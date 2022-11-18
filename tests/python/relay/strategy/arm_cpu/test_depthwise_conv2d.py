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
"""Tests for arm_cpu schedules for depthwise_conv2d."""

from test_generalized_conv2d import GeneralizedConv2dTests
from tvm.testing import fixture, main, parameter, parameters


class DepthwiseConv2dTests(GeneralizedConv2dTests):
    """Helper for constructing depthwise Conv2ds. Sets the reference kernel layout to what x86 code
    supports."""

    @fixture
    def groups(self, data_shape):
        """By definition, a depthwise_conv2d has a number of groups equal to the number of input
        channels, so we don't need to specify the number of groups each time."""
        return data_shape[3]

    def setup_method(self):
        self.ref_kernel_layout = "HWOI"


class TestDepthwiseConv2d_NCHW_OIHW(DepthwiseConv2dTests):
    """This test is for depthwise_conv2d_nchw.arm_cpu schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = parameters(
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 10, 3, 32), (3, 3), 32, 1, 0, 1),
        ((1, 32, 16, 32), (3, 3), 32, 1, (0, 2, 2, 0), 1),
        ((1, 32, 16, 32), (3, 3), 32, 1, 0, 1),
        ((1, 32, 16, 32), (3, 3), 32, 1, 0, 1),
        ((1, 32, 16, 32), (3, 3), 32, 1, (0, 2, 2, 0), 2),
        ((1, 32, 16, 16), (3, 3), 16, 1, (1, 1, 2, 2), 2),
    )

    in_dtype = parameter("int8", "int16")
    data_layout = parameter("NCHW")
    kernel_layout = parameter("OIHW")
    out_layout = parameter("NCHW")
    schedule_name = parameter("depthwise_conv2d_nchw.arm_cpu")


class TestDepthwiseConv2d_NHWC_HWOI(DepthwiseConv2dTests):
    """This test is for depthwise_conv2d_nhwc.generic schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = parameters(
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 10, 16), (3, 3), 16, 1, 0, 1),
        ((1, 49, 10, 64), (10, 4), 64, (2, 1), (4, 1, 5, 1), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 2),
        ((1, 32, 32, 16), (3, 3), 16, 1, (1, 1, 2, 2), 2),
    )

    in_dtype = parameter("int8", "int16")
    data_layout = parameter("NHWC")
    kernel_layout = parameter("HWOI")
    out_layout = parameter("NHWC")
    schedule_name = parameter("depthwise_conv2d_nhwc.generic")


class TestDepthwiseConv2d_NHWC_HWOI_DSP(DepthwiseConv2dTests):
    """This test is for depthwise_conv2d_nhwc_dsp.arm_cpu schedule. The tests that are parameterized
    by dtype work for both int8 and int16, while the others only work on the specified dtype."""

    in_dtype_parameterized_tests = [
        # Depthwise_conv2d parameters from MobileNetV1 0.25x
        ((1, 48, 48, 8), (3, 3), 8, (1, 1), 1),
        ((1, 48, 48, 16), (3, 3), 16, (2, 2), (1, 1, 0, 0)),
        ((1, 24, 24, 32), (3, 3), 32, (1, 1), 1),
        ((1, 24, 24, 32), (3, 3), 32, (2, 2), (1, 1, 0, 0)),
        ((1, 12, 12, 64), (3, 3), 64, (1, 1), 1),
        ((1, 12, 12, 64), (3, 3), 64, (2, 2), (1, 1, 0, 0)),
        ((1, 6, 6, 128), (3, 3), 128, (1, 1), 1),
        ((1, 6, 6, 128), (3, 3), 128, (2, 2), (1, 1, 0, 0)),
        ((1, 3, 3, 256), (3, 3), 256, (1, 1), 1),
        # Asymmetric and larger kernels
        ((1, 25, 5, 64), (3, 3), 64, (1, 1), 1),
        ((1, 24, 24, 8), (5, 5), 8, (1, 1), 1),
        ((1, 24, 24, 8), (3, 5), 8, (1, 1), 1),
    ]

    data_shape, kernel_size, num_filter, strides, padding, in_dtype = parameters(
        # Make a copy of each parameterized test for int8 and one for int16
        *map(lambda t: t + ("int8",), in_dtype_parameterized_tests),
        *map(lambda t: t + ("int16",), in_dtype_parameterized_tests),
        # Test the int16 implementation with channel numbers not divisible by four
        ((1, 48, 48, 6), (3, 3), 6, (1, 1), 1, "int16"),
    )
    dilation = parameter(1)
    data_layout = parameter("NHWC")
    kernel_layout = parameter("HWOI")
    out_layout = parameter("NHWC")
    schedule_name = parameter("depthwise_conv2d_nhwc_dsp.arm_cpu")


class TestDepthwiseConv2d_Tensordot(DepthwiseConv2dTests):
    """This test is for the depthwise_conv2d schedule tensorized using tensordot."""

    data_shape, kernel_size, num_filter, strides, padding, in_dtype = parameters(
        # Currently, our schedule requires kernel_w be divisible by the number of simd lanes given
        # its dtype. This means 3x3 and 5x5 kernels do not work on int16 or int8 for now. If you had
        # to, you could hack around this by padding the data and kernel.
        ((1, 48, 48, 8), (3, 3), 8, (1, 1), 1, "int32"),
        ((1, 48, 48, 16), (3, 3), 16, (2, 2), (1, 1, 0, 0), "int32"),
        ((1, 24, 24, 32), (3, 3), 32, (1, 1), 1, "int32"),
        ((1, 24, 24, 32), (3, 3), 32, (2, 2), (1, 1, 0, 0), "int32"),
        ((1, 12, 12, 64), (3, 3), 64, (1, 1), 1, "int32"),
        ((1, 12, 12, 64), (3, 3), 64, (2, 2), (1, 1, 0, 0), "int32"),
        ((1, 6, 6, 128), (3, 3), 128, (1, 1), 1, "int32"),
        ((1, 6, 6, 128), (3, 3), 128, (2, 2), (1, 1, 0, 0), "int32"),
        ((1, 3, 3, 256), (3, 3), 256, (1, 1), 1, "int32"),
        ((1, 25, 5, 64), (3, 3), 64, (1, 1), 1, "int32"),
        ((1, 24, 24, 8), (5, 5), 8, (1, 1), 1, "int32"),
        ((1, 24, 24, 8), (3, 5), 8, (1, 1), 1, "int32"),
        # These "evenly divisible" kernels work on smaller dtypes.
        ((1, 48, 48, 8), (3, 2), 8, 1, 0, "int16"),
        ((1, 48, 48, 8), (4, 4), 8, 1, 0, "int8"),
    )
    dilation = parameter(1)

    data_layout = parameter("NCHW")
    kernel_layout = parameter("OIHW")
    out_layout = parameter("NHWC", "NCHW")
    schedule_name = parameter("depthwise_conv2d_nchw_oihw_dsp.arm_cpu")


if __name__ == "__main__":
    main()
