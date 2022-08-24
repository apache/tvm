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


ObjectTypes = (ObjectBase, NDArrayBase, Module, ObjectRValueRef, PyNativeObject)


def convert_to_object(value, span=None):
    """Convert a Python value to corresponding object type.

    Parameters
    ----------
    value : str
        The value to be inspected.

    span : Optional[Span]
        The location of this itervar in the source code.

    Returns
    -------
    obj : Object
        The corresponding object value.
    """
    if isinstance(value, ObjectTypes):
        return value
    if isinstance(value, bool):
        return const(value, "uint1x1", span=span)
    if isinstance(value, Number):
        return const(value, span=span)
    if isinstance(value, string_types):
        return _ffi_api.String(value)
    if isinstance(value, (list, tuple)):
        value = [convert_to_object(x) for x in value]
        return _ffi_api.Array(*value)
    if isinstance(value, dict):
        vlist = []
        for item in value.items():
            if (
                not isinstance(item[0], ObjectTypes)
                and not isinstance(item[0], string_types)
                and not isinstance(item[0], Number)
            ):
                raise ValueError("key of map must already been a container type")
            vlist.append(convert_to_object(item[0]))
            vlist.append(convert_to_object(item[1]))
        return _ffi_api.Map(*vlist)
    if isinstance(value, ObjectGeneric):
        return value.asobject()
    if value is None:
        return None

    raise ValueError("don't know how to convert type %s to object" % type(value))


def convert(value, span=None):
    """Convert value to TVM object or function.

    Parameters
    ----------
    value : python value

    span : Optional[Span]
        The location of this statement in the source code.

    Returns
    -------
    tvm_val : Object or Function
        Converted value in TVM
    """
    if isinstance(value, (PackedFuncBase, ObjectBase)):
        return value

    if callable(value):
        return convert_to_tvm_func(value)

    return convert_to_object(value, span=span)


def _scalar_type_inference(value):
    if hasattr(value, "dtype"):
        dtype = str(value.dtype)
    elif isinstance(value, bool):
        dtype = "bool"
    elif isinstance(value, float):
        # We intentionally prefer convert the float to float32 since it's more common in DL.
        if -3.40282347e38 <= value <= 3.40282347e38:
            dtype = "float32"
        else:
            dtype = "float64"
    elif isinstance(value, int):
        # We intentionally prefer convert the python int to int32 since it's more common in DL.
        if -2147483648 <= value <= 2147483647:
            dtype = "int32"
        else:
            dtype = "int64"
    else:
        raise NotImplementedError(
            "Cannot automatically inference the type." " value={}".format(value)
        )
    return dtype


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
