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
import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.testing.aot import AOTTestModel, compile_and_run, generate_ref_data
from tvm.micro.testing.aot_test_utils import (
    AOT_CORSTONE300_RUNNER,
)


class BasicConv2dTests:
    @tvm.testing.requires_corstone300
    def test_depthwise_conv2d_NCHWc(
        self,
        data_shape,
        kernel_size,
        data_layout,
        kernel_layout,
        groups,
        strides,
        padding,
        dilation,
        dtype,
        schedule_name,
    ):
        """Test a subgraph with a single depthwise_conv2d_nchwc operator."""
        ishape = data_shape
        wshape = (data_shape[1], 1, *kernel_size)
        weight_data = np.random.randint(low=-10, high=10, size=wshape, dtype=dtype)
        groups = groups

        input0 = relay.var("input", relay.TensorType(ishape, dtype))
        weight0 = relay.const(weight_data)
        out0 = relay.op.nn.contrib_depthwise_conv2d_nchwc(
            relay.layout_transform(input0, "NCHW", data_layout),
            relay.layout_transform(weight0, "OIHW", kernel_layout),
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            groups=groups,
            out_dtype="",
            out_layout="",
        )
        ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

        input1 = relay.var("input", relay.TensorType(ishape, dtype))
        weight1 = relay.const(weight_data)
        out1 = relay.op.nn.contrib_depthwise_conv2d_nchwc(
            relay.layout_transform(input1, "NCHW", data_layout),
            relay.layout_transform(weight1, "OIHW", kernel_layout),
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            groups=groups,
            out_dtype="",
            out_layout="",
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


class TestDepthWiseConv2d_NCHWc(BasicConv2dTests):
    """This test is for depthwise_conv2d_NCHWc schedule."""

    (
        data_shape,
        kernel_size,
        groups,
        strides,
        padding,
        dilation,
        kernel_layout,
        data_layout,
    ) = tvm.testing.parameters(
        ((1, 16, 32, 32), (3, 3), 16, (1, 1), (1, 1, 1, 1), (1, 1), "OIHW1i4o", "NCHW4c"),
        ((1, 16, 32, 32), (3, 3), 12, (1, 1), (1, 1, 1, 1), (1, 1), "OIHW1i8o", "NCHW8c"),
    )
    dtype = tvm.testing.parameter("int8", "int16", "int32")
    schedule_name = tvm.testing.parameter("depthwise_conv2d_NCHWc")


if __name__ == "__main__":
    tvm.testing.main()
