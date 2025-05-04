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
"""dtype class."""
# pylint: disable=invalid-name
from enum import IntEnum
import numpy as np

from . import core


class DataTypeCode(IntEnum):
    """DataType code in DLTensor."""

    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3
    BFLOAT = 4
    Float8E3M4 = 7
    Float8E4M3 = 8
    Float8E4M3B11FNUZ = 9
    Float8E4M3FN = 10
    Float8E4M3FNUZ = 11
    Float8E5M2 = 12
    Float8E5M2FNUZ = 13
    Float8E8M0FNU = 14
    Float6E2M3FN = 15
    Float6E3M2FN = 16
    Float4E2M1FN = 17


class dtype(str):
    """TVM FFI dtype class.

    Parameters
    ----------
    dtype_str : str

    Note
    ----
    This class subclasses str so it can be directly passed
    into other array api's dtype arguments.
    """

    __slots__ = ["__tvm_ffi_dtype__"]

    NUMPY_DTYPE_TO_STR = {
        np.dtype(np.bool_): "bool",
        np.dtype(np.int8): "int8",
        np.dtype(np.int16): "int16",
        np.dtype(np.int32): "int32",
        np.dtype(np.int64): "int64",
        np.dtype(np.uint8): "uint8",
        np.dtype(np.uint16): "uint16",
        np.dtype(np.uint32): "uint32",
        np.dtype(np.uint64): "uint64",
        np.dtype(np.float16): "float16",
        np.dtype(np.float32): "float32",
        np.dtype(np.float64): "float64",
    }
    if hasattr(np, "float_"):
        NUMPY_DTYPE_TO_STR[np.dtype(np.float_)] = "float64"

    def __new__(cls, content):
        content = str(content)
        val = str.__new__(cls, content)
        val.__tvm_ffi_dtype__ = core.DataType(content)
        return val

    def __repr__(self):
        return f"dtype('{self}')"

    def with_lanes(self, lanes):
        """
        Create a new dtype with the given number of lanes.

        Parameters
        ----------
        lanes : int
            The number of lanes.

        Returns
        -------
        dtype
            The new dtype with the given number of lanes.
        """
        cdtype = core._create_dtype_from_tuple(
            core.DataType, self.__tvm_ffi_dtype__.type_code, self.__tvm_ffi_dtype__.bits, lanes
        )
        val = str.__new__(dtype, str(cdtype))
        val.__tvm_ffi_dtype__ = cdtype
        return val

    @property
    def itemsize(self):
        return self.__tvm_ffi_dtype__.itemsize

    @property
    def type_code(self):
        return self.__tvm_ffi_dtype__.type_code

    @property
    def bits(self):
        return self.__tvm_ffi_dtype__.bits

    @property
    def lanes(self):
        return self.__tvm_ffi_dtype__.lanes


try:
    import ml_dtypes

    dtype.NUMPY_DTYPE_TO_STR[np.dtype(ml_dtypes.bfloat16)] = "bfloat16"
    dtype.NUMPY_DTYPE_TO_STR[np.dtype(ml_dtypes.float8_e4m3fn)] = "float8_e4m3fn"
    dtype.NUMPY_DTYPE_TO_STR[np.dtype(ml_dtypes.float8_e5m2)] = "float8_e5m2"
    dtype.NUMPY_DTYPE_TO_STR[np.dtype(ml_dtypes.float4_e2m1fn)] = "float4_e2m1fn"
except ImportError:
    pass

core._set_class_dtype(dtype)
