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
from tvm.micro.testing.aot_test_utils import AOT_CORSTONE300_RUNNER


class BasicGroupConv2dTests:
    @tvm.testing.requires_corstone300
    def test_conv2d(
        self,
        data_shape,
        data_layout,
        kernel_size,
        kernel_layout,
        num_filter,
        strides,
        padding,
        dilation,
        groups,
        dtype,
        schedule_name,
    ):
        """Test a subgraph with a single conv2d operator."""
        ishape = data_shape

        assert groups > 1, f"groups should be more than 1 to create a group conv2d."

        if data_layout == "NCHW" and kernel_layout == "OIHW":
            assert data_shape[1] % groups == 0
            wshape = (num_filter, data_shape[1] // groups, *kernel_size)
        elif data_layout == "NHWC" and kernel_layout == "HWIO":
            assert data_shape[3] % groups == 0
            wshape = (*kernel_size, data_shape[3] // groups, num_filter)
        else:
            raise ValueError(
                f"Incorrect data layout({data_layout}) and kernel layout({kernel_layout})."
            )

        weight_data = np.random.randint(low=-10, high=10, size=wshape, dtype=dtype)

        input0 = relay.var("input", relay.TensorType(ishape, dtype))
        weight0 = relay.const(weight_data)
        out0 = relay.op.nn.conv2d(
            input0,
            weight0,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
            dilation=(dilation, dilation),
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            out_dtype="int32",
            out_layout=data_layout,
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
            groups=groups,
            dilation=(dilation, dilation),
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            out_dtype="int32",
            out_layout=data_layout,
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


class TestGroupConv2d_NCHW_OIHW(BasicGroupConv2dTests):
    """This test is for group_conv2d_nchw.arm_cpu schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        ((1, 16, 32, 32), (3, 3), 12, 1, 0, 1),
        ((1, 16, 32, 10), (3, 3), 16, 1, 0, 1),
        ((1, 16, 32, 32), (3, 3), 16, 1, (0, 2, 2, 0), 1),
        ((1, 16, 32, 32), (3, 3), 16, 1, 0, 1),
        ((1, 16, 32, 32), (3, 3), 16, 1, 0, 1),
        ((1, 16, 32, 32), (3, 3), 16, 1, (0, 2, 2, 0), 2),
        ((1, 16, 32, 32), (3, 3), 32, 1, (1, 1, 2, 2), 2),
    )
    groups = tvm.testing.parameter(2, 4)
    data_layout = tvm.testing.parameter("NCHW")
    dtype = tvm.testing.parameter("int8", "int16")
    kernel_layout = tvm.testing.parameter("OIHW")
    schedule_name = tvm.testing.parameter("group_conv2d_nchw.arm_cpu")


class TestGroupConv2d_NHWC_HWIO(BasicGroupConv2dTests):
    """This test is for group_conv2d_nhwc.generic schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        ((1, 32, 32, 16), (3, 3), 12, 1, 0, 1),
        ((1, 32, 10, 16), (3, 3), 16, 1, 0, 1),
        ((1, 49, 10, 16), (10, 4), 64, (2, 1), (4, 1, 5, 1), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 2),
        ((1, 32, 32, 16), (3, 3), 16, 1, (1, 1, 2, 2), 2),
    )
    groups = tvm.testing.parameter(2, 4)
    data_layout = tvm.testing.parameter("NHWC")
    dtype = tvm.testing.parameter("int8", "int16")
    kernel_layout = tvm.testing.parameter("HWIO")
    schedule_name = tvm.testing.parameter("group_conv2d_nhwc.generic")


if __name__ == "__main__":
    tvm.testing.main()
