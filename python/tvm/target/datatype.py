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
"""Bring Your Own Datatypes custom datatype framework

TODO(@gussmith23 @hypercubestart) link to BYODT docs when they exist"""
import tvm
from tvm.runtime import convert, DataType
from tvm.tir.expr import (
    Call as _Call,
    Cast as _Cast,
    FloatImm as _FloatImm,
    BinaryOpExpr as _BinaryOpExpr,
)
from tvm.tir.op import call_pure_extern
from tvm._ffi import register_func as _register_func
from tvm.tir import call_intrin


def register(type_name, type_code):
    """Register a custom datatype with the given type name and type code

    Currently, the type code is manually allocated by the user, and the user
    must ensure that no two custom types share the same code. Generally, this
    should be straightforward, as the user will be manually registering all of
    their custom types.

    Example:

    .. code-block:: python

        # Register a dtype named 'posites2' under type code 130.
        tvm.target.datatype.register('posites2', 130)


    Parameters
    ----------
    type_name : str
        The name of the custom datatype.

    type_code : int
        The type's code, which should be >= kCustomBegin. See
        include/tvm/runtime/data_type.h.
    """
    tvm.runtime._ffi_api._datatype_register(type_name, type_code)


def get_type_name(type_code):
    """Get the type name of a custom datatype from the type code.

    Note that this only works for custom datatypes registered with
    tvm.target.datatype.register(). It does not work for TVM-native types.

    Example:

    .. code-block:: python

        tvm.target.datatype.register('posites2', 130)
        assert tvm.target.datatype.get_type_name(130) == 'posites2'

    Parameters
    ----------
    type_code : int
        The type code of the custom datatype.

    Returns
    -------
    type_name : String
        The name of the custom datatype.

    """
    return tvm.runtime._ffi_api._datatype_get_type_name(type_code)


def get_type_code(type_name):
    """Get the type code of a custom datatype from its type name

    Note that this only works for custom datatypes registered with
    tvm.target.datatype.register(). It does not work for TVM-native types.

    Example:

    .. code-block:: python

        tvm.target.datatype.register('posites2', 130)
        assert tvm.target.datatype.get_type_code('posites2') == 130

    Parameters
    ----------
    type_name : str
        The type name

    Returns
    -------
    type_code : int
        The type code of the custom datatype.
    """
    return tvm.runtime._ffi_api._datatype_get_type_code(type_name)


def get_type_registered(type_code):
    """Returns true if a custom datatype is registered under the given type code

    Example:

    .. code-block:: python

        tvm.target.datatype.register('posites2', 130)
        assert tvm.target.datatype.get_type_registered(130)

    Parameters
    ----------
    type_code: int
        The type code

    Returns
    -------
    type_registered : bool
        True if a custom datatype is registered under this type code, and false
        otherwise.
    """
    return tvm.runtime._ffi_api._datatype_get_type_registered(type_code)


def register_op(
    lower_func, op_name, target, src_type_name, dest_type_name=None, intrinsic_name=None
):
    """Register a lowering function for a specific operator of a custom datatype

    At build time, Relay must lower operators over custom datatypes into
    operators it understands how to compile. For each custom datatype operator
    which Relay finds while lowering custom datatypes, Relay expects to find a
    user-defined lowering function. Users register their user-defined lowering
    functions using this function.

    Users should use create_lower_func to create their lowering function. It
    should serve most use-cases.

    Currently, this will work with Casts, intrinsics (e.g. sqrt, sigmoid), and
    binary expressions (e.g. Add, Sub, Mul, Div).

    See the LowerCustomDatatypes pass to see how registered functions are used.

    Lowering Functions
    ------------------
    TODO(@gussmith23) Get the terminology right here.
    Lowering functions take in a Relay node, and should return a semantically
    equivalent Relay node which Relay can build. This means that the returned
    node should not contain any custom datatypes. Users should likely not need
    to define lowering functions by hand -- see the helper function
    create_lower_func.

    Parameters
    ----------
    lower_func : function
        The lowering function to call. See create_lower_func.

    op_name : str
        The name of the operation which the function computes, given by its
        class name (e.g. Add, LE, Cast, Call).

    target : str
        The name of codegen target.

    src_type_name : str
        The name of the custom datatype, e.g. posites2 (but not custom[posites2]32).
        If op_name is not "Cast", then target type is guaranteed to be the same as src_type_name.

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
        lower_func_name = (
            "tvm.datatype.lower."
            + target
            + "."
            + op_name
            + "."
            + dest_type_name
            + "."
            + src_type_name
        )
    elif op_name == "Call" and intrinsic_name is not None:
        lower_func_name = (
            "tvm.datatype.lower."
            + target
            + "."
            + op_name
            + ".intrin."
            + intrinsic_name
            + "."
            + src_type_name
        )
    else:
        lower_func_name = "tvm.datatype.lower." + target + "." + op_name + "." + src_type_name
    tvm._ffi.register_func(lower_func_name, lower_func)


def register_min_func(func, type_name):
    """Register the function that returns the minimum representable value of type_name.

    Operators such as max pooling and argmax require the minimum
    finite value representable by the datatype the op operating on.
    Users can use this function to register a function that returns a TIR expression node
    outputting the minimum representable value of their custom data type.

    Users should use create_min_lower_func to create their lowering function. It
    should serve most use-cases.

    Note: for special cases when it is known that the custom datatype is representable
    by a float, the user can create their own lowering func that returns a FloatImm.
    The benefits are allowing optimizations such as rewrites to work as expected on custom
    datatypes.

    Parameters
    ----------
    func : function
        Input is an integer num_bits, should return a TIR expression node that
        represents a scalar tensor of type custom[type_name]num_bits with the minimum
        representable value.

    type_name : str
        The name of the custom datatype, e.g. posites2 (but not custom[posites2]32).
    """
    _register_func("tvm.datatype.min." + type_name, func)


def create_min_lower_func(extern_func_map, type_name):
    """Returns a lowering function for getting the minimum value of a custom datatype.

    Parameters
    ----------
    extern_func_map : map
        A map from bit lengths to the name of the extern "C" function to lower to.

    type_name : string
        The name of the custom datatype, e.g. posites2 (but not custom[posites2]32).
    """

    def lower(num_bits):
        dtype = f"custom[{type_name}]{num_bits}"

        if num_bits not in extern_func_map:
            raise RuntimeError("missing minimum function for {dtype}")

        return call_pure_extern(dtype, extern_func_map[num_bits])

    return lower


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
        Takes an op---either a Cast, Call, or a binary op (e.g. an Add) and returns a
        call to the specified external function, passing the op's argument
        or arguments. The return type of the call depends
        on the type of the op: if it is a custom type, then a uint of the same
        width as the custom type is returned. Otherwise, the type is
        unchanged."""
        dtype = op.dtype
        t = DataType(dtype)
        if get_type_registered(t.type_code):
            dtype = "uint" + str(t.bits)
            if t.lanes > 1:
                dtype += "x" + str(t.lanes)

        key = t.bits
        if isinstance(op, _Cast):
            src_bits = DataType(op.value.dtype).bits
            key = (src_bits, t.bits)

        if key not in extern_func_map:
            raise RuntimeError(f"missing key {key} in extern_func_map for {op.astext()}")

        if isinstance(op, _Cast):
            return call_pure_extern(dtype, extern_func_map[key], op.value)
        if isinstance(op, _FloatImm):
            return call_pure_extern(dtype, extern_func_map[key], op.value)
        if isinstance(op, _Call):
            return call_pure_extern(dtype, extern_func_map[key], *op.args)
        if isinstance(op, _BinaryOpExpr):
            return call_pure_extern(dtype, extern_func_map[key], op.a, op.b)

        raise RuntimeError(f"lowering unsupported op: {op.astext()}")

    return lower


def lower_ite(ite_op):
    """Lowered if then else function that calls intrinsic if_then_else.
    Unlike a function lowered by create_lower_func, this function
    calls the tvm intrinsic if_then_else.

    Parameters
    ----------
    ite_op : Op
        Takes an if then else op and returns a
        call to tir.if_then_else function, passing the op's
        arguments. The return type of the call if a uint of the same
        width as the custom type is returned.
    """
    dtype = ite_op.dtype
    t = tvm.DataType(dtype)
    assert get_type_registered(t.type_code)
    dtype = "uint" + str(t.bits)
    if t.lanes > 1:
        dtype += "x" + str(t.lanes)
    return call_intrin(
        dtype,
        "tir.if_then_else",
        convert(ite_op.args[0]),
        convert(ite_op.args[1]),
        convert(ite_op.args[2]),
    )


def lower_call_pure_extern(op):
    """Lowered call pure extern function that calls intrinsic call_pure_extern.
    Unlike a function lowered by create_lower_func, this function
    calls the tvm intrinsic call_pure_extern.

    Parameters
    ----------
    ite_op : Op
        Takes a call_pure_extern op and returns a
        call to tir.call_pure_extern function, passing the op's
        arguments. The return type of the call if a uint of the same
        width as the custom type is returned.
    """
    dtype = op.dtype
    t = tvm.DataType(dtype)
    assert get_type_registered(t.type_code)
    dtype = "uint" + str(t.bits)
    if t.lanes > 1:
        dtype += "x" + str(t.lanes)
    return call_intrin(dtype, "tir.call_pure_extern", *op.args)
