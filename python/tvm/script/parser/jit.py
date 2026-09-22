# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Lazy JIT specialization; validation and caches stay outside transpilation."""

import ast
import inspect
from collections.abc import Callable
from typing import Any

from tvm.tirx import PrimFunc

from . import protocol
from .frontend import _definition_scope, _lexical_environment, parse


class OptionalAnnotation:
    """Retain the runtime annotation of a removable JIT parameter."""

    def __init__(self, annotation):
        self.annotation = annotation

    def __tvm_optional_annotation__(self):
        return self.annotation


def make_jit(builder):
    """Create a JIT decorator for the canonical construction namespace."""

    def jit(func=None, *, private=False, check_well_formed=True, is_stir=False, persistent=False):
        def apply(function):
            if not inspect.isfunction(function):
                raise TypeError(f"Expect a function, but got: {function}")
            frame = inspect.currentframe().f_back
            try:
                if frame.f_code is jit.__code__:
                    frame = frame.f_back
                function.__tvm_definition_scope__ = _definition_scope(frame)
            finally:
                del frame
            function.__tvm_function_info__ = jit.__tvm_function_info__
            function.__tvm_function_options__ = {
                "private": private,
                "s_tir": is_stir,
                "persistent": persistent,
            }
            return TIRJit(function, check_well_formed, is_stir, persistent, private)

        return apply(func) if func is not None else apply

    return protocol.register_function(
        jit,
        builder,
        option_map={"private": "private", "is_stir": "s_tir", "persistent": "persistent"},
    )


class TIRJit:
    """Top-level kernel decorator with compile-time ``.specialize()`` params.

    Parses the function body lazily: parsing is deferred until ``.specialize()``
    supplies concrete values for the params annotated as ``T.constexpr``. The
    return type of ``.specialize()`` is a ``tvm.tirx.PrimFunc``, identical in
    type to what ``@T.prim_func`` produces today.

    Constexpr params are removed from the resulting PrimFunc's parameter list;
    their values are baked into the IR (e.g. into ``T.Buffer((M, K), ...)``
    shape annotations and into the body).
    """

    def __init__(
        self,
        func: Callable,
        check_well_formed: bool = True,
        is_stir: bool = False,
        persistent: bool = False,
        private: bool = False,
    ) -> None:
        self.func = func
        self.check_well_formed = check_well_formed
        self.is_stir = is_stir
        self.persistent = persistent  # pylint: disable=unused-private-member
        self.private = private  # pylint: disable=unused-private-member
        # Resolved closure vars (computed once; the function itself is the
        # capture point, so this never changes between specializations).
        self._closure_vars: dict[str, Any] = _lexical_environment(func)
        self._definition_scope = dict(getattr(func, "__tvm_definition_scope__", {}))
        # Detect which params are marked T.constexpr or T.Optional. With PEP 563
        # (``from __future__ import annotations``), each annotation is a
        # string. Resolve only marker names; annotation constructors execute
        # later in the generated declaration's builder context.
        raw_anns = getattr(func, "__annotations__", {}) or {}
        annotation_scope = {**self._closure_vars, **self._definition_scope}

        def resolve_marker(node):
            if isinstance(node, ast.Name):
                return annotation_scope.get(node.id)
            if isinstance(node, ast.Attribute):
                return inspect.getattr_static(resolve_marker(node.value), node.attr, None)
            if isinstance(node, ast.Call) and resolve_marker(node.func) is OptionalAnnotation:
                return OptionalAnnotation
            return None

        sig = inspect.signature(func)
        constexpr_names: set[str] = set()
        constexpr_defaults: dict[str, Any] = {}
        optional_names: set[str] = set()
        for name, param in sig.parameters.items():
            raw_ann = raw_anns.get(name)
            ann = raw_ann
            if isinstance(ann, str):
                try:
                    node = ast.parse(ann, mode="eval").body
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        node = ast.parse(node.value, mode="eval").body
                    ann = resolve_marker(node)
                except SyntaxError:
                    ann = None
            if protocol.is_constexpr_marker(ann):
                constexpr_names.add(name)
                if param.default is not inspect.Parameter.empty:
                    constexpr_defaults[name] = param.default
            if ann is OptionalAnnotation or isinstance(ann, OptionalAnnotation):
                optional_names.add(name)
        self.constexpr_names: frozenset[str] = frozenset(constexpr_names)
        self.constexpr_defaults: dict[str, Any] = constexpr_defaults
        self.optional_names: frozenset[str] = frozenset(optional_names)
        self._cache: dict[tuple, PrimFunc] = {}

    def specialize(self, **specialization_kwargs) -> PrimFunc:
        """Build a PrimFunc by binding constexprs and absent optional params.

        Parameters
        ----------
        **specialization_kwargs
            One value per ``T.constexpr``-annotated parameter.  A
            ``T.Optional`` parameter may additionally be supplied as ``None``
            to remove it from the resulting PrimFunc ABI.  Omitting an
            optional parameter keeps it as a normal runtime parameter.

        Returns
        -------
        PrimFunc
            A concrete TIRx PrimFunc, identical in type to the output of
            ``@T.prim_func``.
        """
        specializable_names = self.constexpr_names | self.optional_names
        extra = specialization_kwargs.keys() - specializable_names
        if extra:
            raise TypeError(
                f"{self.func.__name__}.specialize() got unexpected arg(s): "
                f"{sorted(extra)} (specializable params are: {sorted(specializable_names)})"
            )
        invalid_optional = {
            name: specialization_kwargs[name]
            for name in self.optional_names & specialization_kwargs.keys()
            if specialization_kwargs[name] is not None
        }
        if invalid_optional:
            raise TypeError(
                f"{self.func.__name__}.specialize(): T.Optional parameters only accept None; "
                f"pass actual tensors when calling the compiled kernel (got: {invalid_optional!r})"
            )

        supplied_constexprs = {
            name: specialization_kwargs[name]
            for name in self.constexpr_names & specialization_kwargs.keys()
        }
        effective = {**self.constexpr_defaults, **supplied_constexprs}
        missing = self.constexpr_names - effective.keys()
        if missing:
            raise TypeError(
                f"{self.func.__name__}.specialize() missing constexpr arg(s) "
                f"(no default provided): {sorted(missing)}"
            )

        absent_params = {name: None for name in self.optional_names & specialization_kwargs.keys()}
        try:
            cache_key = (
                tuple((name, type(value), value) for name, value in sorted(effective.items())),
                tuple(sorted(absent_params)),
            )
            cached = self._cache.get(cache_key)
        except TypeError as err:
            raise TypeError(
                f"{self.func.__name__}.specialize(): all constexpr values must "
                f"be hashable (got: {effective!r})"
            ) from err
        if cached is not None:
            return cached

        prim_func = parse(
            self.func,
            self._closure_vars,
            _definition_scope=self._definition_scope,
            _specialization_bindings={**effective, **absent_params},
        )
        if self.check_well_formed:
            from tvm.s_tir.analysis import verify_well_formed
            from tvm.tirx.analysis import verify_tirx_well_formed

            verify_well_formed(prim_func)
            if not prim_func.attrs.get("s_tir", False):
                verify_tirx_well_formed(prim_func)
        setattr(prim_func, "__name__", self.func.__name__)
        self._cache[cache_key] = prim_func
        return prim_func
