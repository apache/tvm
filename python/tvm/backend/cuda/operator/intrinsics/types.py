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
"""PTX data types for CUDA codegen."""

import enum

import tvm_ffi

from_string_func = tvm_ffi.get_global_func("tirx.intrinsics.cuda.PTXDTypeFromString")
to_string_func = tvm_ffi.get_global_func("tirx.intrinsics.cuda.PTXDTypeToString")


class PTXDataType(enum.Enum):
    """
    A Python equivalent of the provided C++ DataType enum class.

    Inherits from IntEnum so that members behave both as enum members
    and as integers, mirroring the C++ behavior.

    see also src/target/source/ptx.cc
    """

    INT4 = 0
    UINT4 = 1
    INT8 = 2
    UINT8 = 3
    INT16 = 4
    UINT16 = 5
    INT32 = 6
    UINT32 = 7
    INT64 = 8
    UINT64 = 9
    FLOAT4_E2M1FN = 10
    FLOAT6_E2M3FN = 11
    FLOAT6_E3M2FN = 12
    FLOAT8_E4M3FN = 13
    FLOAT8_E4M3FNUZ = 14
    FLOAT8_E5M2 = 15
    FLOAT8_E8M0FNU = 16
    FLOAT16 = 17
    BFLOAT16 = 18
    FLOAT16X2 = 19
    FLOAT32 = 20
    TENSOR_FLOAT32 = 21
    FLOAT64 = 22
    BIT1 = 23
    BIT8 = 24
    BIT16 = 25
    BIT32 = 26
    BIT64 = 27

    @classmethod
    def from_string(cls, s_type: str) -> "PTXDataType":
        return PTXDataType(from_string_func(s_type))

    def to_string(self) -> str:
        return to_string_func(self.value)
