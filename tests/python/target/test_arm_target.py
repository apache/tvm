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
import pytest

import tvm
from tvm.topi.arm_cpu.conv2d_int8 import is_int8_hw_support
from tvm.target import codegen

llvm_version, arm_target, input_dtype, kernel_dtype, is_supported = tvm.testing.parameters(
    # Testing mcpu type
    (8, "c -mcpu=cortex-m4", "int8", "int8", False),
    (8, "c -mcpu=cortex-m7", "int8", "int8", False),
    (8, "c -mcpu=cortex-m33", "int8", "int8", False),
    (8, "c -mcpu=cortex-m55", "int8", "int8", False),
    (8, "c -mcpu=cortex-m3", "int8", "int8", False),
    (7, "llvm -mtriple=arm-linux-gnueabi -mattr=+neon", "int8", "int8", False),
    (8, "llvm -mtriple=arm-linux-gnueabi -mattr=+neon", "int8", "int8", True),
    (9, "llvm -mtriple=arm-linux-gnueabi -mattr=+neon", "int8", "int8", True),
    (8, "llvm -mtriple=arm-linux-gnueabi", "int8", "int8", False),
    (7, "llvm -mtriple=aarch64-linux-gnu -mattr=+v8.4a,+dotprod", "int8", "int8", False),
    (8, "llvm -mtriple=aarch64-linux-gnu -mattr=+v8.4a,+dotprod", "int8", "int8", True),
    (9, "llvm -mtriple=arm-linux-gnueabi -mattr=+neon", "int8", "int8", True),
    (8, "llvm -mtriple=aarch64-linux-gnu", "int8", "int8", True),
    # Testing dtype
    (8, "llvm -mtriple=aarch64-linux-gnu -mattr=+neon", "int16", "int8", False),
    (8, "llvm -mtriple=aarch64-linux-gnu -mattr=+neon", "int8", "int16", False),
    (8, "llvm -mtriple=aarch64-linux-gnu -mattr=+neon", "int16", "int16", False),
)


def test_arm_conv2d_int8_support(
    monkeypatch, llvm_version, arm_target, input_dtype, kernel_dtype, is_supported
):
    """Test ARM conv2d int8 support for different targets.

    Parameters
    ----------
    arm_target : str
        ARM CPU target.
    input_dtype : str
        Conv2d input data type.
    kernel_dtype : Session
        Conv2d kernel data type.
    is_supported : bool
        Expected result.
    """
    with tvm.target.Target(arm_target):
        monkeypatch.setattr(codegen, "llvm_version_major", lambda: llvm_version)
        assert is_int8_hw_support(input_dtype, kernel_dtype) == is_supported
