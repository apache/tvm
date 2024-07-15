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
"""Tests for Arm(R) A-Profile Architecture."""
import os

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relay
from tvm.relay.transform import ToMixedPrecision, FoldConstant
from tvm.relay.build_module import bind_params_by_name


def get_mattr(dtype):
    mattr = "+v8.2a,+neon"
    if dtype == "float16":
        mattr += ",+fullfp16"
    elif dtype == "bfloat16":
        mattr += ",+bf16"
    return mattr


@tvm.testing.skip_if_32bit(reason="skipping test for i386.")
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_conv2d(dtype):
    """Test if Conv2d cross compiles with TVM schedules."""
    dtype = "float32"
    ishape = [1, 28, 28, 3]  # NHWC
    kernel_size = (3, 3)
    wshape = (kernel_size[0], kernel_size[1], ishape[-1], 2)  # HWIO
    weight_data = np.random.uniform(-128, 127, wshape).astype(dtype)
    invar = relay.var("data", relay.TensorType(ishape, dtype))
    weight = relay.const(weight_data, dtype)
    out = relay.op.nn.conv2d(
        invar,
        weight,
        kernel_size=kernel_size,
        channels=2,
        strides=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype=dtype,
        out_layout="NHWC",
    )
    mod = tvm.IRModule.from_expr(relay.Function([invar], out))
    params = {}

    prefixed_network_name = dtype + ".conv2d"
    lib_path = os.getcwd() + "/" + prefixed_network_name + ".mod.so"
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=" + get_mattr(dtype)

    mod["main"] = bind_params_by_name(mod["main"], params)
    if dtype in ["float16", "bfloat16"]:
        mod = ToMixedPrecision(dtype)(mod)
        mod = FoldConstant()(mod)

    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(mod, target=target, params=params)
        lib.export_library(lib_path, cc="aarch64-linux-gnu-gcc")


if __name__ == "__main__":
    tvm.testing.main()
