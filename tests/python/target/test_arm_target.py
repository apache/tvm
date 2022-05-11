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

arm_target, input_dtype, kernel_dtype, is_supported = tvm.testing.parameters(
    # Testing mcpu type
    ("c -mcpu=cortex-m4 -keys=arm_cpu", "int8", "int8", False),
    ("c -mcpu=cortex-m7 -keys=arm_cpu", "int8", "int8", False),
    ("c -mcpu=cortex-m33 -keys=arm_cpu", "int8", "int8", False),
    ("c -mcpu=cortex-m55 -keys=arm_cpu", "int8", "int8", False),
    ("c -mcpu=cortex-m3 -keys=arm_cpu", "int8", "int8", False),
    ("llvm -keys=arm_cpu -mattr=+neon", "int8", "int8", True),
    # This fails because of a bug in topi.arm_cpu.arm_utils.get_arch_version
    # ("llvm -keys=arm_cpu -mattr=v8.4a,+dotprod", "int8", "int8", True),
    # Testing dtype
    ("llvm -keys=arm_cpu -mattr=+neon", "int16", "int8", False),
    ("llvm -keys=arm_cpu -mattr=+neon", "int8", "int16", False),
    ("llvm -keys=arm_cpu -mattr=+neon", "int16", "int16", False),
)


def test_arm_conv2d_int8_support(arm_target, input_dtype, kernel_dtype, is_supported):
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
        expected_result = is_supported and (codegen.llvm_version_major() >= 8)
        assert is_int8_hw_support(input_dtype, kernel_dtype) == expected_result
