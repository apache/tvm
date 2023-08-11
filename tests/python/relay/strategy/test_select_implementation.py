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

""" Tests strategy selection for Relay ops """
import pytest
import tvm
from tvm import relay
from tvm import te
from tvm.relay.testing import run_infer_type
import tvm.testing


@pytest.mark.parametrize(
    "target, expected_implementation",
    [("llvm", "concatenate.cpu"), ("llvm -device=arm_cpu", "concatenate.arm_cpu")],
)
def test_concatenate(target, expected_implementation):
    target = tvm.target.Target(target)

    shape = (1, 1, 1, 3)
    dtype = "float32"
    axis = 1
    inputs = []
    inputs.append(relay.var("var0", shape=shape, dtype=dtype))
    inputs.append(relay.var("var1", shape=shape, dtype=dtype))
    input_tuple = relay.Tuple(inputs)
    out = relay.op.concatenate(input_tuple, axis)
    out = run_infer_type(out)

    impl, xx = relay.backend.te_compiler.select_implementation(
        relay.op.get("concatenate"),
        out.attrs,
        [te.placeholder(shape)],
        out.checked_type,
        target,
        use_autotvm=False,
    )
    assert impl.name == expected_implementation


@pytest.mark.parametrize(
    "target,expected_impl",
    [
        ("llvm -device=arm_cpu", "conv2d_nhwc_spatial_pack.arm_cpu"),
        (
            "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon",
            "conv2d_NHWC_quantized_interleaved.arm_cpu",
        ),
        (
            "llvm -device=arm_cpu -mtriple=armv8l-linux-gnu -mattr=+neon",
            "conv2d_nhwc_spatial_pack.arm_cpu",
        ),
    ],
)
def test_int8_conv2d(target, expected_impl):
    target = tvm.target.Target(target)

    dtype = "int8"
    data_shape = (1, 1, 1, 4)
    weight_shape = (1, 1, 4, 4)
    data_layout = "NHWC"
    kernel_layout = "HWIO"
    channels = 4
    kernel_size = (1, 1)

    out = relay.nn.conv2d(
        relay.var("data", shape=data_shape, dtype=dtype),
        relay.var("weight", shape=weight_shape, dtype=dtype),
        kernel_size=kernel_size,
        channels=channels,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )
    out = run_infer_type(out)

    with target:
        impl, _ = relay.backend.te_compiler.select_implementation(
            out.op,
            out.attrs,
            [te.placeholder(data_shape, dtype), te.placeholder(weight_shape, dtype)],
            out.checked_type,
            target,
        )

    assert impl.name == expected_impl


@pytest.mark.parametrize(
    "target,expected_impl",
    [
        ("llvm -device=arm_cpu", "depthwise_conv2d_nhwc.generic"),
        (
            "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon",
            "depthwise_conv2d_nhwc.arm_cpu",
        ),
        (
            "llvm -device=arm_cpu -mtriple=armv8l-linux-gnu -mattr=+neon",
            "depthwise_conv2d_nhwc.generic",
        ),
        ("c -device=arm_cpu -mcpu=cortex-m55", "depthwise_conv2d_nhwc_dsp.arm_cpu"),
    ],
)
def test_int8_depthwise_conv2d(target, expected_impl):
    target = tvm.target.Target(target)

    dtype = "int8"
    out_dtype = "int32"
    data_shape = (2, 2, 4, 8)
    weight_shape = (2, 2, 8, 1)
    data_layout = "NHWC"
    kernel_layout = "HWOI"
    groups = 8
    kernel_size = (2, 2)

    out = relay.nn.conv2d(
        relay.var("data", shape=data_shape, dtype=dtype),
        relay.var("weight", shape=weight_shape, dtype=dtype),
        kernel_size=kernel_size,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        groups=groups,
        out_dtype=out_dtype,
    )
    out = run_infer_type(out)

    with target:
        impl, _ = relay.backend.te_compiler.select_implementation(
            out.op,
            out.attrs,
            [te.placeholder(data_shape, dtype), te.placeholder(weight_shape, dtype)],
            out.checked_type,
            target,
        )

    assert impl.name == expected_impl


if __name__ == "__main__":
    tvm.testing.main()
