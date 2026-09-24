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
"""Static syntax policies keyed by registered builder namespace paths.

Each language variant registers its namespace through ``register_namespace`` and supplies
explicit canonical keys such as ``T.int32`` or ``R.Tensor`` at its marker sites.
The first registered alias names that namespace. Source aliases normalize to this
same root; nested namespace members append their attribute names. Consumers read
the public dictionaries directly. Ordinary captured callables, local aliases,
instance methods and descriptors do not acquire syntax policies through identity
or receiver inference.

Tables retain static syntax facts only, never source functions, captures, frames
or constructed results. Registration decorators return the same callable, except
that ``args_policy`` preserves its existing builder-owned eager annotation adapter.
Shared builder operations are documented in
``tvm.script.ir_builder.parser_protocol``; concrete language variants own registration
and namespace initialization.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from inspect import signature
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, NoReturn, TypeVar

_Callable = TypeVar("_Callable", bound=Callable[..., Any])


class ExprStrPolicy(NamedTuple):
    """Expression-string fields, symbolic dtype and bare-string interpretation.

    Nested strings in marked fields always represent expressions. ``scalar_strings``
    controls bare strings. The dtype is forwarded to native symbol resolution;
    this static record contains no resolved symbols or construction state.
    """

    fields: tuple[str, ...]
    dtype: object = None
    scalar_strings: bool = True


class ArgsPolicy(NamedTuple):
    """Argument policies and positional names computed once at registration.

    ``fields`` maps parameter names to ``expr_str`` or ``global_info``.
    ``expression`` describes the expression-string subset. ``positional_parameters``
    lists positional-only and positional-or-keyword names in signature order;
    keyword-only parameters remain available through ``fields``.
    """

    fields: Mapping[str, str]
    expression: ExprStrPolicy
    positional_parameters: tuple[str, ...]


ARGS_POLICIES: dict[str, ArgsPolicy] = {}
TYPE_VAR_DECL: dict[str, object] = {}
MUTABLE_CELL_DECL: dict[str, frozenset[str]] = {}
RESULT_SPAN: dict[str, bool] = {}
MODULE_DECORATOR: dict[str, bool] = {}
DECLARATION_KIND: dict[str, Literal["function", "helper"]] = {}


def constexpr(value: object) -> NoReturn:
    """Mark host control syntax or a JIT specialization annotation.

    Parameters
    ----------
    value : object
        Source expression to evaluate with ordinary Python semantics in a
        supported control-flow position. The marker itself can also appear
        as an annotation identifying a JIT specialization parameter.

    Raises
    ------
    TypeError
        If invoked directly instead of being recognized in parsed source.

    Notes
    -----
    The parser recognizes this marker through its fixed namespace path and
    removes it before execution. Host operators retain ordinary Python behavior.
    Direct invocation raises TypeError; no builder frame or IR is constructed.

    .. code:: python

        # Source
        if I.constexpr(enabled):
            T.evaluate(1)
        # Builder
        if enabled:
            X.emit_(X.evaluate(1))
    """
    raise TypeError("constexpr is a parser syntax marker, not a runtime operation")


def args_policy(
    namespace_path: str,
    fields: Mapping[str, str],
    *,
    scalar_strings: bool = True,
    dtype: object = None,
    as_type: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register argument syntax at an explicit canonical namespace path.

    Parameters
    ----------
    namespace_path : str
        Registered namespace alias followed by the exported member path.
    fields : Mapping[str, str]
        Parameter names mapped to ``expr_str`` or ``global_info``.
    scalar_strings : bool, optional
        Whether bare strings in expression fields denote expressions. Defaults to True.
    dtype : object, optional
        Opaque dtype passed to the language variant's symbol resolver. None (the default)
        leaves dtype selection to that resolver.
    as_type : bool, optional
        Preserve the eager annotation-class surface, including Python type unions.
        Defaults to False.

    Returns
    -------
    Callable
        Decorator retaining the existing builder-owned annotation adapter where
        needed. Concrete arguments still invoke the original constructor.

    Notes
    -----
    Registration validates policy kinds and parameter names, and inspects the
    signature once. Static policy and positional names are reused for every call.
    Outside construction, unresolved expression annotations yield MissingType;
    inside construction, unresolved strings raise TypeError and TypeVars resolve
    through the active function frame. Those eager semantics belong to the builder.

    .. code:: python

        @args_policy("M.Tensor", {"shape": "expr_str"})
        def Tensor(shape):
            ...
        # Source: M.Tensor(("n",))
        # Builder: M.Tensor((M.resolve_type_var_("n"),))
    """
    fields = dict(fields)
    unsupported = set(fields.values()).difference(("expr_str", "global_info"))
    if unsupported:
        raise ValueError(f"Unknown argument policies: {sorted(unsupported)}")

    def decorate(constructor: Callable[..., Any]) -> Callable[..., Any]:
        call_signature = signature(constructor)
        unknown = set(fields).difference(call_signature.parameters)
        if unknown:
            raise ValueError(f"Unknown argument policy fields: {sorted(unknown)}")
        positional_parameters = tuple(
            parameter.name
            for parameter in call_signature.parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        )
        expression_fields = tuple(name for name, kind in fields.items() if kind == "expr_str")
        expression = ExprStrPolicy(expression_fields, dtype, bool(scalar_strings))
        if expression_fields or as_type:
            from tvm.script.ir_builder.base import wrap_expression_constructor

            result = wrap_expression_constructor(
                constructor, call_signature, expression, as_type=as_type
            )
        else:
            result = constructor
        ARGS_POLICIES[namespace_path] = ArgsPolicy(
            MappingProxyType(fields.copy()), expression, positional_parameters
        )
        return result

    return decorate


def handle_call_args_policy(
    node: ast.Call, resolve: Callable[[ast.expr], str | None]
) -> tuple[ArgsPolicy, tuple[str, ...]] | None:
    """Select a fixed namespace call's policy before visiting its arguments.

    ``resolve`` returns only canonical namespace paths, without evaluating a
    callee or receiver. Unregistered calls return None. Matching calls reuse the
    positional names stored at registration; expression-string rewriting remains
    in ``expr_str_handling`` and the normal call visitor traverses children once.
    """
    namespace_path = resolve(node.func)
    if namespace_path is None:
        return None
    policy = ARGS_POLICIES.get(namespace_path)
    return (policy, policy.positional_parameters) if policy is not None else None


def register_type_var_decl(
    namespace_path: str,
    constructor: _Callable,
    *,
    dtype: object = None,
) -> _Callable:
    """Register symbolic declaration syntax and return the unchanged constructor.

    Parameters
    ----------
    namespace_path : str
        Canonical registered namespace alias and exported callable path, such
        as ``"T.int32"``. A later registration at this path replaces its dtype.
    constructor : Callable
        Callable providing the eager construction operation. Registration
        records its syntax path without invoking or wrapping this callable.
    dtype : object, optional
        Static scalar dtype forwarded to the language variant's symbol resolver.
        None (the default) leaves the dtype unspecified for that resolver.

    Returns
    -------
    Callable
        The exact ``constructor`` object.

    Notes
    -----
    Zero-argument calls denote declarations. Dictionary membership distinguishes
    an unspecified dtype from an unregistered
    constructor. The parser predeclares symbols needed by signatures while the
    native function owns their identity. No constructor executes at registration.

    .. code:: python

        register_type_var_decl("T.int32", T.int32, dtype="int32")
        # Source: n = T.int32()
        # Builder: n = X.resolve_type_var_("n", dtype="int32")
    """
    TYPE_VAR_DECL[namespace_path] = dtype
    return constructor


def mutable_cell_decl(
    namespace_path: str, *, syntax: str = "call"
) -> Callable[[_Callable], _Callable]:
    """Register persistent mutable storage syntax at a fixed namespace path.

    Parameters
    ----------
    namespace_path : str
        Canonical registered namespace alias and exported constructor path.
    syntax : str, optional
        Declaration form: ``"call"`` (the default) for a constructor call,
        ``"annotation"`` for an annotated local declaration, or ``"parameter"``
        for a function parameter annotation. Repeated registration adds forms
        for the same path without removing existing ones.

    Returns
    -------
    Callable
        Registration decorator that returns its constructor unchanged.

    Raises
    ------
    ValueError
        If ``syntax`` is not one of the three supported forms.

    Notes
    -----
    Explicit declarations take precedence over mutable target updates;
    storage creation and updates stay outside ordinary ``bind_``.

    .. code:: python

        mutable_cell_decl("T.int32", syntax="annotation")(T.int32)
        # Source: x: T.int32 = 0
        # Builder: x = X.decl_mutable_cell_(0, ty=X.int32, name="x")
    """
    if syntax not in ("call", "annotation", "parameter"):
        raise ValueError("Mutable declaration syntax must be call, annotation or parameter")

    def decorate(constructor: _Callable) -> _Callable:
        MUTABLE_CELL_DECL[namespace_path] = MUTABLE_CELL_DECL.get(
            namespace_path, frozenset()
        ) | frozenset((syntax,))
        return constructor

    return decorate


def result_span(namespace_path: str) -> Callable[[_Callable], _Callable]:
    """Declare that a call's complete IR effect is represented by its result.

    Parameters
    ----------
    namespace_path : str
        Canonical registered namespace alias and exported callable path whose
        returned object carries the call's complete IR effect.

    Returns
    -------
    Callable
        Registration decorator that returns its callable unchanged.

    Notes
    -----
    The unchanged callable owns no unrelated emitted statements needing caller
    context. This permits result attachment instead of an opaque-call context;
    argument instrumentation and ordinary binding/emission remain. Assignment
    RHS locations pass separately to the language variant's ``bind_``.

    .. code:: python

        @result_span("X.make_node")
        def make_node(value):
            return Node(value)
        # Nested source: X.make_node(x)
        # Builder: _S[i](X.make_node(x))
    """

    def decorate(constructor: _Callable) -> _Callable:
        RESULT_SPAN[namespace_path] = True
        return constructor

    return decorate


def module_decorator(namespace_path: str) -> Callable[[_Callable], _Callable]:
    """Mark a module entry point and return the same callable.

    Parameters
    ----------
    namespace_path : str
        Canonical registered namespace alias and exported module-decorator path,
        such as ``"I.ir_module"``.

    Returns
    -------
    Callable
        Registration decorator that returns the module entry point unchanged.

    Notes
    -----
    Member function decorators use this fixed namespace syntax to defer parsing
    until the complete enclosing module is available. No source-function records,
    captured scopes or constructed module results are retained in the table.

    .. code:: python

        @module_decorator("I.ir_module")
        def ir_module(source):
            return parse(source)
    """

    def decorate(decorator: _Callable) -> _Callable:
        MODULE_DECORATOR[namespace_path] = True
        return decorator

    return decorate


def declaration_kind(
    namespace_path: str, kind: Literal["function", "helper"]
) -> Callable[[_Callable], _Callable]:
    """Classify a fixed namespace decorator's declaration syntax.

    Parameters
    ----------
    namespace_path : str
        Canonical registered namespace alias and exported source-decorator path.
    kind : {"function", "helper"}
        ``"function"`` declares an IR function. ``"helper"`` retains ordinary
        Python helper semantics, including macro expansion and ``I.pyfunc``.
        A later registration at the same path replaces its declaration kind.

    Returns
    -------
    Callable
        Registration decorator that returns its source decorator unchanged.

    Notes
    -----
    Unregistered decorators have neither declaration kind.
    Concrete namespaces register their entry points where they expose them::

        prim_func = declaration_kind("T.prim_func", "function")(make_decorator(builder))
        inline = declaration_kind("T.inline", "helper")(make_macro_decorator(builder))

    The table stores only the kind string. The unchanged decorator supplies its
    builder and options through ordinary entry arguments; no builders, option
    defaults, source functions, captures or callable identities are retained.
    """

    def decorate(decorator: _Callable) -> _Callable:
        DECLARATION_KIND[namespace_path] = kind
        return decorator

    return decorate
