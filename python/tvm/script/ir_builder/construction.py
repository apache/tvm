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
"""Runtime owners for generated builder programs."""

from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from typing import TypeVar

import tvm_ffi

from tvm import ir
from tvm.script.parser.protocol import is_constexpr_marker

from . import ir as I
from .base import MISSING, IRBuilder, source_span
from .parser_support import TypeVarFrame
from .type_var_frame import TypeVarDecl

# Process-wide owner-supplied callbacks, initialized unset and replaced only by
# explicit entry-module registration. Neither table contains per-parse state.
_OPAQUE_FACTORY = None
_SPECIALIZATION = ContextVar("tvm_builder_specialization", default=None)
_ABSENT_PARAMETERS = ContextVar("tvm_builder_absent_parameters", default=None)


@contextmanager
def specialization_context(name, bindings):
    """Pass validated JIT bindings to one root builder execution."""
    token = _SPECIALIZATION.set((name, dict(bindings)))
    try:
        yield
    finally:
        _SPECIALIZATION.reset(token)


@contextmanager
def absent_parameters(name, parameters):
    """Transport the JIT's explicit absent-parameter map to execution.

    JIT owns argument validation/defaults/cache. This input path only carries
    previously selected absences for a single root function, and never infers
    absence from ordinary lexical captures.
    """
    parameters = dict(parameters or {})
    if any(value is not None for value in parameters.values()):
        raise TypeError("absent_params values must be None")
    token = _ABSENT_PARAMETERS.set((name, parameters))
    try:
        yield
    finally:
        _ABSENT_PARAMETERS.reset(token)


class _FunctionReference(ir.GlobalVar):
    """Keep a module-owned GlobalVar callable without changing its native object."""

    def __init__(self, reference):
        # The FFI constructor convention retains the returned existing handle;
        # it does not create, clone, or replace the underlying native GlobalVar.
        # The temporary identity callback owns no registration or persistent state.
        self.__init_handle_by_constructor__(tvm_ffi.convert_func(lambda: reference))

    def asobject(self):
        """Return the original native reference without copying or changing it."""
        return self

    def __call__(self, *args):
        # Imports are deferred until execution to avoid the builder namespaces'
        # initialization cycle. Only the nearest function owns call semantics;
        # module/symbol/block frames do not select a dialect.
        if IRBuilder.is_in_scope():
            from tvm.relax.script.builder.frame import FunctionFrame
            from tvm.tirx.script.builder.frame import PrimFuncFrame

            for frame in reversed(IRBuilder.current().frames):
                if isinstance(frame, PrimFuncFrame):
                    from tvm.tirx.script.builder.ir import _call_global

                    return _call_global(self, *args)
                if isinstance(frame, FunctionFrame):
                    from tvm import relax

                    return relax.Call(self, [relax.utils.convert_to_expr(arg) for arg in args])
        return ir.GlobalVar.__call__(self, *args)


def _function_reference(value):
    """Expose a module GlobalVar as a callable while preserving other values."""
    if isinstance(value, ir.GlobalVar) and not isinstance(value, _FunctionReference):
        return _FunctionReference(value)
    return value


def capture_value(value):
    """Retain lexical values, adapting native function references uniformly."""
    return _function_reference(value)


def register_opaque_factory(factory):
    """Register the constructor for opaque Python module functions."""
    global _OPAQUE_FACTORY
    if not callable(factory):
        raise TypeError("The module construction callback must be callable")
    _OPAQUE_FACTORY = factory


class FunctionRecord:
    """Own one function's declaration and definition construction state."""

    def __init__(
        self,
        builder,
        name,
        options,
        location=None,
        *,
        local=False,
        captures=None,
        parameters=(),
        capture_names=(),
    ):
        self.builder, self.name, self.options = builder, name, dict(options)
        self.span, self.local = source_span(location), local
        self.symbols = TypeVarFrame()
        # Explicit lexical primitive symbols are captures, not new declarations.
        # Parameter names shadow them; all concrete inspection stays builder-side.
        for captured_name in capture_names:
            value = (captures or {}).get(captured_name)
            if (
                captured_name not in parameters
                and ir.is_prim_var(value)
                and value.name in ("", captured_name)
            ):
                self.symbols.bind(captured_name, value)
        context = _SPECIALIZATION.get()
        self.specialization = (
            context[1] if context is not None and context[0] == name and not local else {}
        )
        self.captured_bindings = dict(captures or {}) if not local else {}
        absent = _ABSENT_PARAMETERS.get()
        self.absent_parameters = (
            absent[1] if absent is not None and absent[0] == name and not local else {}
        )
        self.params = {}
        self.runtime_params = {}
        self.signature_symbols = set(self.symbols.symbols)
        self.return_type = MISSING
        self.reference = self.function = None

    @contextmanager
    def declaration(self):
        """Enter the retained symbol frame and a native declaration frame."""
        mode = {"local": True} if self.local else {}
        with self.symbols:
            with self.builder.decl_function(**self.options, **mode, span=self.span) as frame:
                self.builder.func_name(self.name)
                yield self
        self.reference = _function_reference(frame.reference)

    def parameter_lazy(self, name, annotation, location=None):
        """Resolve a declaration at execution time, omitting specialized ABI slots."""
        if name in self.specialization:
            value = self.specialization[name]
            self.params[name] = value
            return value
        if name in self.absent_parameters:
            self.params[name] = None
            return None
        annotation = annotation()
        if is_constexpr_marker(annotation):
            if name not in self.captured_bindings:
                raise TypeError(f"constexpr parameter {name!r} requires a specialization binding")
            value = self.captured_bindings[name]
            self.params[name] = value
            return value
        # Optional is an annotation wrapper, owned by the JIT entry point.
        unwrap = getattr(annotation, "__tvm_optional_annotation__", None)
        if unwrap is not None:
            annotation = unwrap()
        return self.parameter(name, annotation, location)

    def parameter(self, name, annotation, location=None):
        """Construct and retain a parameter in the active declaration."""
        if callable(annotation) and not isinstance(annotation, ir.Expr | ir.Type):
            annotation = annotation()
        if isinstance(annotation, TypeVarDecl):
            annotation = self.symbols.resolve(name, annotation.ty, span=source_span(location))
        if isinstance(annotation, ir.PrimType):
            annotation = self.symbols.resolve(name, annotation, span=source_span(location))
        elif ir.is_prim_var(annotation):
            annotation = self.symbols.bind(name, annotation)
        value = self.builder.arg(name, annotation, span=source_span(location))
        self.params[name] = value
        self.runtime_params[name] = value
        self.signature_symbols = set(self.symbols.symbols)
        return value

    def predeclare(self, name, annotation, location=None):
        """Reserve an explicit symbol type before dependent signature shapes."""
        if callable(annotation) and not isinstance(annotation, ir.Expr | ir.Type):
            annotation = annotation()
        if isinstance(annotation, TypeVarDecl):
            annotation = annotation.ty
        if ir.is_prim_var(annotation):
            value = self.symbols.bind(name, annotation)
        else:
            value = self.symbols.resolve(name, annotation, span=source_span(location))
        # A body declaration is already an explicit binding, even in a function
        # with no parameters. Only symbols newly introduced by the return
        # annotation itself are unbound return-only symbols.
        self.signature_symbols = set(self.symbols.symbols)
        return value

    def symbol(self, name):
        """Read a declared symbol after signature construction."""
        return self.symbols.symbols[name]

    def capture(self, name, fallback):
        """Resolve a free lexical name against retained signature symbols."""
        if isinstance(fallback, TypeVar):
            return self.symbols.resolve(name)
        return self.symbols.symbols.get(name, _function_reference(fallback))

    def returns(self, annotation):
        """Set the return annotation on the record and active native frame."""
        introduced = set(self.symbols.symbols) - self.signature_symbols
        if introduced:
            raise ValueError(
                f"Return annotation introduces unbound symbol {sorted(introduced)[0]!r}"
            )
        self.return_type = annotation
        self.builder.func_ret_type(annotation)

    def define(self, body):
        """Run a generated body with retained parameters under a definition frame."""
        mode = {"local": True, "reference": self.reference} if self.local else {}
        with self.symbols:
            with self.builder.function(**self.options, **mode, span=self.span) as frame:
                self.builder.func_name(self.name)
                for name, value in self.runtime_params.items():
                    self.builder.arg(name, value)
                if self.return_type is not MISSING:
                    self.builder.func_ret_type(self.return_type)
                body(**self.params)
        self.function = frame.function
        self.function.__name__ = self.name
        return self.function


class ModuleProgram:
    """Own module construction while a generated builder program executes."""

    def __init__(self, name=None, original=None, bases=()):
        self.name, self.original, self.bases = name, original, bases
        self.builder = IRBuilder()
        self.frame = None
        self.namespace = SimpleNamespace()
        self.python_functions = {}
        self.result = None

    def __enter__(self):
        self.builder.__enter__()
        self.frame = I.ir_module()
        self.frame.__enter__()
        return self

    def __exit__(self, *error):
        try:
            self.frame.__exit__(*error)
            if error[0] is None:
                self.result = self.builder.get()
                if self.python_functions:
                    self.result.pyfuncs = self.python_functions
                if self.name is not None:
                    self.result.__name__ = self.name
        finally:
            self.builder.__exit__(*error)

    def reserve(self, name):
        """Reserve and expose a stable named module function reference."""
        reference = _function_reference(I.reserve_function(name))
        setattr(self.namespace, name, reference)
        return reference

    def member(self, name, value):
        """Register a class assignment in the generated module namespace."""
        if isinstance(value, ir.BaseFunc):
            reference = I.decl_function(name, value)
            I.def_function(name, value)
            value = reference
        value = _function_reference(value)
        setattr(self.namespace, name, value)
        return value

    def python(self, name, function, source, location=None):
        """Register a host function without executing its body."""
        if _OPAQUE_FACTORY is None:
            raise ValueError("No opaque Python function constructor has been registered")
        opaque = _OPAQUE_FACTORY(name, function, source, source_span(location))
        reference = I.decl_function(name, opaque)
        I.def_function(name, opaque)
        self.python_functions[name] = function
        reference = _function_reference(reference)
        setattr(self.namespace, name, reference)
        return reference


def require_defined(value, name):
    """Return an exported value or report an undefined source name."""
    if value is MISSING:
        raise NameError(f"name {name!r} is not defined")
    return value


def is_python_bool(value):
    """Check for a host boolean without invoking truth conversion."""
    return isinstance(value, bool)
