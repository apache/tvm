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


class BasicDepthwiseConv2dTests:
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
        dtype,
        schedule_name,
    ):
        """Test a subgraph with a single conv2d operator."""
        ishape = data_shape
        groups = num_filter

        assert groups > 1, f"groups should be more than 1 to create a depthwise conv2d."

        if data_layout == "NCHW" and kernel_layout == "OIHW":
            assert (
                num_filter == data_shape[1]
            ), f"Output channels({num_filter}) should be equal to input channels({data_shape[1]})."
            wshape = (num_filter, data_shape[1] // groups, *kernel_size)
        elif data_layout == "NHWC" and kernel_layout == "HWOI":
            assert (
                num_filter == data_shape[3]
            ), f"Output channels({num_filter}) should be equal to input channels({data_shape[3]})."
            wshape = (*kernel_size, num_filter, data_shape[3] // groups)
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


class TestDepthwiseConv2d_NCHW_OIHW(BasicDepthwiseConv2dTests):
    """This test is for depthwise_conv2d_nchw.arm_cpu schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        ((1, 16, 32, 32), (3, 3), 16, 1, 0, 1),
        ((1, 32, 10, 3), (3, 3), 32, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 32, 1, (0, 2, 2, 0), 1),
        ((1, 32, 32, 16), (3, 3), 32, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 32, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 32, 1, (0, 2, 2, 0), 2),
        ((1, 16, 32, 16), (3, 3), 16, 1, (1, 1, 2, 2), 2),
    )
    data_layout = tvm.testing.parameter("NCHW")
    dtype = tvm.testing.parameter("int8", "int16")
    kernel_layout = tvm.testing.parameter("OIHW")
    schedule_name = tvm.testing.parameter("depthwise_conv2d_nchw.arm_cpu")


class TestDepthwiseConv2d_NHWC_HWOI(BasicDepthwiseConv2dTests):
    """This test is for depthwise_conv2d_nhwc.generic schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 10, 16), (3, 3), 16, 1, 0, 1),
        ((1, 49, 10, 64), (10, 4), 64, (2, 1), (4, 1, 5, 1), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 2),
        ((1, 32, 32, 16), (3, 3), 16, 1, (1, 1, 2, 2), 2),
    )
    data_layout = tvm.testing.parameter("NHWC")
    dtype = tvm.testing.parameter("int8", "int16")
    kernel_layout = tvm.testing.parameter("HWOI")
    schedule_name = tvm.testing.parameter("depthwise_conv2d_nhwc.generic")


class TestDepthwiseConv2d_NHWC_HWOI_DSP(BasicDepthwiseConv2dTests):
    """This test is for depthwise_conv2d_nhwc_dsp.arm_cpu schedule."""

    # Tests that work with both int8 and int16 data types. Tuple elements are:
    # data_shape, kernel_size, num_filter, strides, padding
    dtype_parameterized_tests = [
        # Depthwise_conv2d parameters from MobileNetV1 0.25x. The LLVM implementation doesn't
        # support "SAME" and "VALID" padding, so padding must be explicitly specified.
        ((1, 48, 48, 8), (3, 3), 8, (1, 1), 1),
        ((1, 48, 48, 16), (3, 3), 16, (2, 2), (1, 1, 0, 0)),
        ((1, 24, 24, 32), (3, 3), 32, (1, 1), 1),
        ((1, 24, 24, 32), (3, 3), 32, (2, 2), (1, 1, 0, 0)),
        ((1, 12, 12, 64), (3, 3), 64, (1, 1), 1),
        ((1, 12, 12, 64), (3, 3), 64, (2, 2), (1, 1, 0, 0)),
        ((1, 6, 6, 128), (3, 3), 128, (1, 1), 1),
        ((1, 6, 6, 128), (3, 3), 128, (2, 2), (1, 1, 0, 0)),
        ((1, 3, 3, 256), (3, 3), 256, (1, 1), 1),
        # Asymmetric height and width
        ((1, 25, 5, 64), (3, 3), 64, (1, 1), 1),
        # Larger kernel
        ((1, 24, 24, 8), (5, 5), 8, (1, 1), 1),
        # Asymmetric kernel
        ((1, 24, 24, 8), (3, 5), 8, (1, 1), 1),
    ]

    data_shape, kernel_size, num_filter, strides, padding, dtype = tvm.testing.parameters(
        # Make a copy of each parameterized test for int8 and one for int16
        *map(lambda t: t + ("int8",), dtype_parameterized_tests),
        *map(lambda t: t + ("int16",), dtype_parameterized_tests),
        # Test the int16 implementation with channel numbers not divisible by four
        ((1, 48, 48, 6), (3, 3), 6, (1, 1), 1, "int16"),
    )
    dilation = tvm.testing.parameter(1)
    data_layout = tvm.testing.parameter("NHWC")
    kernel_layout = tvm.testing.parameter("HWOI")
    schedule_name = tvm.testing.parameter("depthwise_conv2d_nhwc_dsp.arm_cpu")


class TestDepthwiseConv2d_Tensordot(BasicDepthwiseConv2dTests):
    data_shape, kernel_size, num_filter, strides, padding, dtype = tvm.testing.parameters(
        # Currently, our schedule requires kernel_w be divisible by the number of simd lanes given
        # its dtype. This means 3x3 and 5x5 kernels do not work on int16 or int8 for now. If you had
        # to, you could hack around this by padding the data and kernel.
        ((1, 8, 48, 48), (3, 3), 8, (1, 1), 1, "int32"),
        ((1, 16, 48, 48), (3, 3), 16, (2, 2), (1, 1, 0, 0), "int32"),
        ((1, 32, 24, 24), (3, 3), 32, (1, 1), 1, "int32"),
        ((1, 32, 24, 24), (3, 3), 32, (2, 2), (1, 1, 0, 0), "int32"),
        ((1, 64, 12, 12), (3, 3), 64, (1, 1), 1, "int32"),
        ((1, 64, 12, 12), (3, 3), 64, (2, 2), (1, 1, 0, 0), "int32"),
        ((1, 128, 6, 6), (3, 3), 128, (1, 1), 1, "int32"),
        ((1, 128, 6, 6), (3, 3), 128, (2, 2), (1, 1, 0, 0), "int32"),
        ((1, 256, 3, 3), (3, 3), 256, (1, 1), 1, "int32"),
        ((1, 64, 25, 5), (3, 3), 64, (1, 1), 1, "int32"),
        ((1, 8, 24, 24), (5, 5), 8, (1, 1), 1, "int32"),
        ((1, 8, 24, 24), (3, 5), 8, (1, 1), 1, "int32"),
        # These "evenly divisible" kernels work on smaller dtypes.
        ((1, 8, 48, 48), (3, 2), 8, 1, 0, "int16"),
        ((1, 8, 48, 48), (4, 4), 8, 1, 0, "int8"),
    )
    dilation = tvm.testing.parameter(1)
    data_layout = tvm.testing.parameter("NCHW")
    kernel_layout = tvm.testing.parameter("OIHW")
    schedule_name = tvm.testing.parameter("depthwise_conv2d_nchw_oihw_dsp.arm_cpu")


if __name__ == "__main__":
    tvm.testing.main()
