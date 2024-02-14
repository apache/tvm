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
"""Common implementation of object generic related logic"""
# pylint: disable=unused-import, invalid-name
from numbers import Number, Integral
from tvm._ffi.base import string_types
from tvm._ffi.runtime_ctypes import ObjectRValueRef

from . import _ffi_node_api, _ffi_api
from .object import ObjectBase, PyNativeObject, _set_class_object_generic
from .ndarray import NDArrayBase
from .packed_func import PackedFuncBase, convert_to_tvm_func
from .module import Module


class ObjectGeneric(object):
    """Base class for all classes that can be converted to object."""

    def asobject(self):
        """Convert value to object"""
        raise NotImplementedError()


ObjectTypes = (ObjectBase, NDArrayBase, Module, ObjectRValueRef, PackedFuncBase, PyNativeObject)


def convert_to_object(value):
    """Convert a Python value to corresponding object type.

    Type conversions performed by this function must *only* produce
    types that are supported by `libtvm_runtime.so`.  This function
    must be usable in environments where only TVM runtime support is
    present.  Automatic conversions to compile-time representations
    (e.g. `tir.IntImm` or `relax.PrimValue`) should not be done as
    part of this conversion, as these types are not available in
    `libtvm_runtime.so`.

    Parameters
    ----------
    value : str
        The value to be inspected.

    Returns
    -------
    obj : Object
        The corresponding object value.

    """

    if isinstance(value, ObjectTypes):
        return value
    elif isinstance(value, (bool, int, float)):
        return value
    elif isinstance(value, string_types):
        return _ffi_api.String(value)
    elif isinstance(value, (list, tuple)):
        # The call to _ffi_api.Array will convert its own arguments,
        # so we don't need to apply any explicit conversions here.
        return _ffi_api.Array(*value)
    elif isinstance(value, dict):
        if any(not isinstance(key, (ObjectTypes, string_types, Number)) for key in value):
            raise ValueError("key of map must already been a container type")

        vlist = [kv for item in value.items() for kv in item]
        return _ffi_api.Map(*vlist)
    elif isinstance(value, ObjectGeneric):
        return value.asobject()
    elif callable(value):
        return convert_to_tvm_func(value)
    elif value is None:
        return None
    else:
        raise TypeError(f"don't know how to convert type {type(value)} to object")


def convert(value):
    """Convert value to TVM object or function.

    Parameters
    ----------
    value : python value

    Returns
    -------
    tvm_val : Object or Function
        Converted value in TVM

    Note
    ----
    This function is redirected to `convert_to_object` as it is widely used in
    the codebase. We can choose one to keep and discard the other one later.
    """

    return convert_to_object(value)


def _scalar_type_inference(value):
    if hasattr(value, "dtype"):
        return str(value.dtype)
    elif isinstance(value, bool):
        return "bool"
    elif isinstance(value, float):
        # We intentionally prefer convert the float to float32 since it's more common in DL.
        if -3.40282347e38 <= value <= 3.40282347e38:
            return "float32"
        else:
            return "float64"
    elif isinstance(value, int):
        # We intentionally prefer convert the python int to int32 since it's more common in DL.
        if -2147483648 <= value <= 2147483647:
            return "int32"
        else:
            return "int64"
    else:
        raise NotImplementedError(f"Cannot automatically inference the type. value={value}")


def const(value, dtype=None, span=None):
    """construct a constant

    Parameters
    ----------
    value : number
        The content of the constant number.

    dtype : str or None, optional
        The data type.

    span : Optional[Span]
        The location of the constant value in the source.

    Returns
    -------
    const_val: tvm.Expr
        The result expression.
    """
    if dtype is None:
        dtype = _scalar_type_inference(value)
    if dtype == "uint64" and value >= (1 << 63):
        return _ffi_node_api.LargeUIntImm(dtype, value & ((1 << 32) - 1), value >> 32, span)
    return _ffi_node_api._const(value, dtype, span)


_set_class_object_generic(ObjectGeneric, convert_to_object)
