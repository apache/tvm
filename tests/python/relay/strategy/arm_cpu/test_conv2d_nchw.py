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
import sys
import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import relay
from tvm.testing.aot import AOTTestModel, compile_and_run, generate_ref_data
from tvm.micro.testing.aot_test_utils import (
    AOT_CORSTONE300_RUNNER,
)


class BasicConv2dTests:
    @tvm.testing.requires_corstone300
    def test_conv2d(
        self,
        data_shape,
        kernel_size,
        kernel_layout,
        num_filter,
        strides,
        padding,
        dilation,
        dtype,
        schedule_name,
    ):
        """Test a subgraph with a single conv2d_nchw operator."""
        ishape = data_shape
        wshape = (num_filter, data_shape[1], *kernel_size)
        weight_data = np.random.randint(low=-10, high=10, size=wshape, dtype=dtype)

        input0 = relay.var("input", relay.TensorType(ishape, dtype))
        weight0 = relay.const(weight_data)
        out0 = relay.op.nn.conv2d(
            input0,
            weight0,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=(dilation, dilation),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
            out_layout="NCHW",
        )
        ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

        input1 = relay.var("input", relay.TensorType(ishape, dtype))
        weight1 = relay.const(weight_data)

        out1 = relay.op.nn.conv2d(
            input1,
            weight1,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=(dilation, dilation),
            data_layout="NCHW",
            kernel_layout=kernel_layout,
            out_dtype="int32",
            out_layout="NCHW",
        )
        mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

        inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
        output_list = generate_ref_data(ref_mod, inputs)

        compile_and_run(
            AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
            runner=AOT_CORSTONE300_RUNNER,
            interface_api="c",
            use_unpacked_api=True,
            target_opts={
                "-keys": "arm_cpu",
                "-mcpu": "cortex-m7",
            },
            schedule_name=schedule_name,
        )


class TestConv2d_OIHW_small_kernel(BasicConv2dTests):
    """This test is for conv2d_nchw_spatial_pack.arm_cpu schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation, dtype = tvm.testing.parameters(
        ((1, 16, 32, 32), (3, 3), 12, 1, 0, 1, "int8"),
        ((1, 16, 32, 32), (3, 3), 12, 1, 0, 1, "int16"),
        ((1, 32, 16, 16), (3, 3), 12, 1, 0, 1, "int16"),
    )
    kernel_layout = tvm.testing.parameter("OIHW")
    schedule_name = tvm.testing.parameter("conv2d_nchw_spatial_pack.arm_cpu")


if __name__ == "__main__":
    tvm.testing.main()
