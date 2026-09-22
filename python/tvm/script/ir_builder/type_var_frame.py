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
"""Function-owned symbolic type variables on the native builder frame stack."""

from dataclasses import dataclass
from functools import wraps
from typing import TypeVar

import tvm_ffi

from tvm import ir

from . import _ffi_api
from .base import IRBuilder, IRBuilderFrame, source_span


@dataclass(frozen=True)
class TypeVarDecl:
    """Describe an unbound primitive symbolic declaration."""

    ty: object

    def __post_init__(self):
        ty = ir.PrimType(self.ty) if isinstance(self.ty, str) else self.ty
        if not isinstance(ty, ir.PrimType):
            raise TypeError("A symbolic declaration requires a primitive type")
        object.__setattr__(self, "ty", ty)


@tvm_ffi.register_object("script.ir_builder.TypeVarFrame")
class TypeVarFrame(IRBuilderFrame):
    """Retain canonical symbols across one function declaration and definition."""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.TypeVarFrame)

    @staticmethod
    def current():
        """Find the nearest active function symbol frame."""
        for frame in reversed(IRBuilder.current().frames):
            if isinstance(frame, TypeVarFrame):
                return frame
        raise ValueError("Symbol resolution requires an active TypeVarFrame")

    def resolve(self, name, ty=None, *, span=None):
        """Resolve or introduce one canonical primitive symbol."""
        if not isinstance(name, str) or not name:
            raise ValueError("A symbolic variable requires a nonempty string name")
        if isinstance(ty, str):
            ty = ir.PrimType(ty)
        if ty is not None and not isinstance(ty, ir.PrimType):
            raise TypeError("A symbolic variable requires a primitive type")
        if name in self.symbols:
            return self.symbols[name]
        value = ir.Var(name, ir.PrimType("int64") if ty is None else ty, source_span(span))
        _ffi_api.TypeVarFrameSetSymbol(self, name, value)
        return value

    def bind(self, name, symbol):
        """Register an existing primitive variable or reuse its canonical binding."""
        if not isinstance(name, str) or not name:
            raise ValueError("A symbolic variable requires a nonempty string name")
        if not ir.is_prim_var(symbol):
            raise TypeError("A symbolic binding requires a primitive Var")
        if name in self.symbols:
            return self.resolve(name, symbol.ty)
        if not symbol.name:
            IRBuilder.name(name, symbol)
        _ffi_api.TypeVarFrameSetSymbol(self, name, symbol)
        return symbol


def resolve_type_var(name, *, dtype=None, span=None):
    """Resolve a symbol in the nearest active function frame."""
    return TypeVarFrame.current().resolve(name, dtype, span=span)


def wrap_expression_constructor(constructor, call_signature, policy, *, as_type=False):
    """Adapt an eager constructor using parser-owned expression-string metadata."""
    fields = policy.fields

    def unresolved(value, nested=False):
        if isinstance(value, str):
            return nested or policy.scalar_strings
        if isinstance(value, TypeVar):
            return True
        if isinstance(value, tuple | list):
            return any(unresolved(item, True) for item in value)
        return False

    @wraps(constructor)
    def invoke(*args, **kwargs):
        bound = call_signature.bind(*args, **kwargs)
        if IRBuilder.is_in_scope():
            # typing.TypeVar is ordinary eager Python metadata. Resolve it
            # here, never in the syntax-only transpiler.
            def resolve(value):
                if isinstance(value, TypeVar):
                    if value.__bound__ is not None or value.__constraints__:
                        raise TypeError("A symbolic TypeVar cannot have constraints or a bound")
                    return TypeVarFrame.current().resolve(value.__name__)
                if isinstance(value, tuple):
                    return tuple(resolve(item) for item in value)
                if isinstance(value, list):
                    return [resolve(item) for item in value]
                return value

            for field in fields:
                if field in bound.arguments:
                    bound.arguments[field] = resolve(bound.arguments[field])
        if any(unresolved(bound.arguments[field]) for field in fields if field in bound.arguments):
            if IRBuilder.is_in_scope():
                raise TypeError(
                    "Builder expression arguments require concrete symbols, not strings"
                )
            return ir.Type.missing()
        return constructor(*bound.args, **bound.kwargs)

    result = invoke
    if as_type:
        # The class is an annotation surface, not an IR or proxy type.
        # __new__ returns the concrete construction result (or MissingType).
        result = type(
            constructor.__name__,
            (),
            {
                "__new__": lambda cls, *args, **kwargs: invoke(*args, **kwargs),
                "__signature__": call_signature,
                "__doc__": constructor.__doc__,
                "__module__": constructor.__module__,
            },
        )
    return result
