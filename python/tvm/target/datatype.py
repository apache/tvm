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
"""Custom datatype functionality"""
import tvm
from tvm.runtime import convert, DataType
from tvm.tir.expr import (Call as _Call, Cast as _Cast,
                          FloatImm as _FloatImm, BinaryOpExpr as _BinaryOpExpr)
from tvm.tir.op import call_pure_extern
from tvm._ffi import register_func as _register_func
from tvm.tir import call_intrin


def register(type_name, type_code):
    """Register a custom datatype with the given type name and type code
    Currently, the type code is manually allocated by the user, and the
    user must ensure that no two custom types share the same code.
    Generally, this should be straightforward, as the user will be
    manually registering all of their custom types.

    Parameters
    ----------
    type_name : str
        The name of the custom datatype

    type_code : int
        The type's code, which should be >= kCustomBegin
    """
    tvm.runtime._ffi_api._datatype_register(type_name, type_code)


def get_type_name(type_code):
    """Get the type name from the type code

    Parameters
    ----------
    type_code : int
        The type code
    """
    return tvm.runtime._ffi_api._datatype_get_type_name(type_code)


def get_type_code(type_name):
    """Get the type code from the type name

    Parameters
    ----------
    type_name : str
        The type name
    """
    return tvm.runtime._ffi_api._datatype_get_type_code(type_name)


def get_type_registered(type_code):
    """Get a boolean representing whether the type is registered

    Parameters
    ----------
    type_code: int
        The type code
    """
    return tvm.runtime._ffi_api._datatype_get_type_registered(type_code)


def register_op(lower_func,
                op_name,
                target,
                src_type_name,
                dest_type_name=None,
                intrinsic_name=None):
    """Register an external function which computes the given op.

    Currently, this will only work with Casts and binary expressions
    whose arguments are named `a` and `b`.
    TODO(gus) figure out what other special cases must be handled by
        looking through expr.py.

    Parameters
    ----------
    lower_func : function
        The lowering function to call. See create_lower_func.

    op_name : str
        The name of the operation which the function computes, given by its
        class name (e.g. Add, LE, Cast).

    target : str
        The name of codegen target.

    src_type_name : str
        The name of the custom datatype, e.g. posit (but not custom[posit]8).

    dest_type_name : str
        If op_name is "Cast", then this is required and should be set to the dest datatype of
        the argument to the Cast. If op_name is not "Cast", this is unused.

    intrinsic_name : str
        If op_name is "Call" and intrinsic_name is not None, then we assume the
        op is a Call to an Intrinsic, and intrinsic_name is the intrinsic's
        name.
    """

    if op_name == "Cast":
        assert dest_type_name is not None
        lower_func_name = "tvm.datatype.lower." + target + "." + op_name + "." \
                          + dest_type_name + "." + src_type_name
    elif op_name == "Call" and intrinsic_name is not None:
        lower_func_name = "tvm.datatype.lower." + target + "." + op_name \
                          + ".intrin." + intrinsic_name + "." + src_type_name
    else:
        lower_func_name = "tvm.datatype.lower." + target + "." + op_name + "." \
                          + src_type_name
    tvm._ffi.register_func(lower_func_name, lower_func)

# TODO(gus) could probably make this a decorator if i want
def register_min_func(func, type_name):
    _register_func("tvm.datatype.min." + type_name, func)

def create_lower_func(extern_func_map):
    """Returns a function which lowers an operation to a function call.

    Parameters
    ----------
    extern_func_map : map
        If lowering a Cast, extern_func_map should be a map from tuples of
        (src_bit_length, dest_bit_length) to the name of the extern "C" function to lower to.

        Otherwise, for unary and binary ops, it should simply be a map
        from bit_length to the name of the extern "C" function to lower to.
    """
    def lower(op):
        """
        Takes an op---either a Cast or a binary op (e.g. an Add) and returns a
        call to the specified external function, passing the op's argument
        (Cast) or arguments (a binary op). The return type of the call depends
        on the type of the op: if it is a custom type, then a uint of the same
        width as the custom type is returned. Otherwise, the type is
        unchanged."""
        dtype = op.dtype
        t = DataType(dtype)
        if get_type_registered(t.type_code):
            dtype = "uint" + str(t.bits)
            if t.lanes > 1:
                dtype += "x" + str(t.lanes)
        if isinstance(op, _Cast):
            src_bits = bit_length(op.value.dtype)
            return call_pure_extern(dtype, extern_func_map[(src_bits, t.bits)], op.value)
        if isinstance(op, _FloatImm):
            return call_pure_extern(dtype, extern_func_map[t.bits], op.value)
        if isinstance(op, _Call):
            return call_pure_extern(dtype, extern_func_map[t.bits], *op.args)
        if isinstance(op, _BinaryOpExpr):
            return call_pure_extern(dtype, extern_func_map[t.bits], op.a, op.b)

        raise RuntimeError(f"lowering unsupported op: {op.astext()}")

    return lower

def bit_length(type_str):
    t = DataType(type_str)
    return t.bits

def lower_ite(ite_intrin):
    dtype = ite_intrin.dtype
    t = tvm.DataType(dtype)
    assert get_type_registered(t.type_code)
    dtype = "uint" + str(t.bits)
    if t.lanes > 1:
        dtype += "x" + str(t.lanes)
    return call_intrin(dtype, "tir.if_then_else", convert(ite_intrin.args[0]),
                       convert(ite_intrin.args[1]),
                       convert(ite_intrin.args[2]))
