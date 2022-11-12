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
"""TVM Script Parser Typing Class for TIR

This module provides typing class for TVM script type annotation usage, it can be viewed as
a wrapper for uniform Type system in IR
"""
# pylint: disable=invalid-name
from numbers import Integral

import tvm
from .special_stmt import SpecialStmt, convert_to_int


class TypeGeneric:  # pylint: disable=too-few-public-methods
    """Base class for all the TVM script typing class"""

    def evaluate(self):
        """Return an actual ir.Type Object that this Generic class wraps"""
        raise TypeError("Cannot get tvm.Type from a generic type")

    def require_type_generic_at(self, idx):  # pylint: disable=unused-argument
        """If True, the `idx`th type argument must be TypeGeneric"""
        return True

    # This function is added here to avoid a pylint error
    # for T.int/float below not being callable
    def __call__(self):
        raise NotImplementedError()


class ConcreteType(TypeGeneric):  # pylint: disable=too-few-public-methods, abstract-method
    """TVM script typing class for uniform Type objects

    Params
    ------
    vtype: Union[str, tvm.ir.Type]

        The IR type represented by the type annotation.  If a string
        (e.g. "float32"), this represents a `ir.PrimType` generated
        from that string.  If a `ir.Type` is provided, this represents
        the type provided.
    """

    def __init__(self, vtype):
        if isinstance(vtype, tvm.ir.Type):
            self.type = vtype
        else:
            self.type = tvm.ir.PrimType(vtype)

    def __call__(self, *args):  # pylint: disable=arguments-differ
        pass

    def evaluate(self):
        return self.type


class VoidType(ConcreteType):  # pylint: disable=too-few-public-methods, abstract-method
    """TVM script typing class for void type"""

    def __init__(self):
        super().__init__("")


class GenericPtrType(TypeGeneric):  # pylint: disable=abstract-method
    """TVM script typing class generator for PtrType

    [] operator is overloaded, accepts a ConcreteType and an optional storage scope string,
    returns a ConcreteType wrapping PtrType
    """

    def __getitem__(self, args):
        if isinstance(args, TypeGeneric):
            args = [args]
        if len(args) == 1:
            vtype, scope = args[0], "global"
        elif len(args) == 2:
            vtype, scope = args[0], args[1]
        else:
            raise TypeError(f"Illegal type argument num for Ptr")
        if not isinstance(vtype, TypeGeneric):
            raise TypeError(f"Ptr expects a type argument, but received {type(vtype).__name__}")
        if not isinstance(scope, str):
            raise TypeError(f"Ptr expects storage scope argument be a string")
        return ConcreteType(tvm.ir.PointerType(vtype.evaluate(), scope))

    def require_type_generic_at(self, idx):
        return idx != 1  # the second argument is storage scope for Ptr


class GenericTupleType(TypeGeneric):  # pylint: disable=abstract-method
    """TVM script typing class generator for TupleType

    [] operator is overloaded, accepts a list of ConcreteType and returns a ConcreteType
    wrapping TupleType
    """

    def __getitem__(self, vtypes):
        if isinstance(vtypes, TypeGeneric):
            vtypes = [vtypes]
        return ConcreteType(tvm.ir.TupleType([vtype.evaluate() for vtype in vtypes]))


class GenericBufferType(SpecialStmt):  # pylint: disable=too-few-public-methods, abstract-method
    """TVM script typing class for uniform Type objects"""

    def __init__(self, vtype):
        def match_buffer_syntax_sugar(
            shape,
            dtype: str = "float32",
            name: str = None,
            data=None,
            strides=None,
            elem_offset=None,
            scope="global",
            align=-1,
            offset_factor=0,
            buffer_type="default",
            axis_separators=None,
            span=None,
        ):
            if strides is None:
                strides = []
            align = convert_to_int(align, "align", self.context.report_error, self.node.span)
            offset_factor = convert_to_int(
                offset_factor, "offset_factor", self.context.report_error, self.node.span
            )
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                name,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
                axis_separators,
                span=span,
            )
            return buffer

        self.type = vtype
        super().__init__(match_buffer_syntax_sugar, def_symbol=True)

    def __call__(
        self,
        shape,
        dtype="float32",
        *,
        name: str = None,
        data=None,
        strides=None,
        elem_offset=None,
        scope="global",
        align=-1,
        offset_factor=0,
        buffer_type="default",
        axis_separators=None,
        span=None,
    ):
        """
        This function is for Buffer(...) syntax sugar.
        """
        pass  # pylint: disable=unnecessary-pass

    def __getitem__(self, args):
        """
        This function is for Buffer[...] syntax sugar
        Note that args is the list of all arguments
        """
        if len(args) < 2:
            raise ValueError("T.Buffer[...] needs at least two arguments: shape and dtype.")

        shape = args[0]
        dtype = args[1]

        valid_shape = isinstance(shape, (tvm.ir.PrimExpr, Integral, tuple, list))
        valid_dtype = isinstance(dtype, str)
        if not (valid_shape and valid_dtype):
            raise ValueError(
                "The first argument of T.Buffer[...] needs to be a tuple, "
                "followed by the second argument dtype as a string"
            )


# add all floating point and integer datatypes to the module
for _dtype in ["float", "uint", "int"]:
    for _size in ["8", "16", "32", "64"]:
        for _lanes in ["", "x4", "x8", "x16", "x32", "x64"]:
            _name = _dtype + _size + _lanes
            globals()[_name] = ConcreteType(_name)


# All other DataType annotations are represented with the same string
# as is used by `tvm.runtime.DataType`.  This does redefine the Python
# built-in bool, but only within the context of `tvm.script.tir.ty`
# and `tvm.script.tir` modules.  The `T.boolean` alias is maintained
# for backwards compatibility.

bool = ConcreteType("bool")  # pylint: disable=redefined-builtin
boolean = bool


handle = ConcreteType("handle")
void = VoidType()
Ptr = GenericPtrType()
Tuple = GenericTupleType()
# we don't have 'buffer' type on the cpp side
# thus 'handle' is used here for convenience's sake
Buffer = GenericBufferType("handle")
