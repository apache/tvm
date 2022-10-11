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

import tvm.testing
from test_generalized_conv2d import GeneralizedConv2dTests

class TestDepthwiseConv2d_NCHW_OIHW(GeneralizedConv2dTests):
    """This test is for depthwise_conv2d_nchw.arm_cpu schedule."""

    data_shape, groups, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        ((1, 32, 32, 16), 16, (3, 3), 16, 1, 0, 1),
        ((1, 10, 3, 32), 32, (3, 3), 32, 1, 0, 1),
        ((1, 32, 16, 32), 32, (3, 3), 32, 1, (0, 2, 2, 0), 1),
        ((1, 32, 16, 32), 32, (3, 3), 32, 1, 0, 1),
        ((1, 32, 16, 32), 32, (3, 3), 32, 1, 0, 1),
        ((1, 32, 16, 32), 32, (3, 3), 32, 1, (0, 2, 2, 0), 2),
        ((1, 32, 16, 16), 16, (3, 3), 16, 1, (1, 1, 2, 2), 2),
    )
    in_dtype = tvm.testing.parameter("int8", "int16")
    data_layout, kernel_layout, out_layout = tvm.testing.parameters(("NCHW", "OIHW", "NCHW"),)
    schedule_name = tvm.testing.parameter("depthwise_conv2d_nchw.arm_cpu")


class TestDepthwiseConv2d_NHWC_HWOI(GeneralizedConv2dTests):
    """This test is for depthwise_conv2d_nhwc.generic schedule."""

    data_shape, groups, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        ((1, 32, 32, 16), 16, (3, 3), 16, 1, 0, 1),
        ((1, 32, 10, 16), 16, (3, 3), 16, 1, 0, 1),
        ((1, 49, 10, 64), 64, (10, 4), 64, (2, 1), (4, 1, 5, 1), 1),
        ((1, 32, 32, 16), 16, (3, 3), 16, 1, (0, 2, 2, 0), 1),
        ((1, 32, 32, 16), 16, (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), 16, (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), 16, (3, 3), 16, 1, (0, 2, 2, 0), 2),
        ((1, 32, 32, 16), 16, (3, 3), 16, 1, (1, 1, 2, 2), 2),
    )
    in_dtype = tvm.testing.parameter("int8", "int16")
    data_layout, kernel_layout, out_layout = tvm.testing.parameters(("NHWC", "HWOI", "NHWC"),)
    schedule_name = tvm.testing.parameter("depthwise_conv2d_nhwc.generic")


class TestDepthwiseConv2d_NHWC_HWOI_DSP(GeneralizedConv2dTests):
    """This test is for depthwise_conv2d_nhwc_dsp.arm_cpu schedule."""

    # Tests that work with both int8 and int16 data types. Tuple elements are:
    # data_shape, kernel_size, num_filter, strides, padding
    in_dtype_parameterized_tests = [
        # Depthwise_conv2d parameters from MobileNetV1 0.25x. The LLVM implementation doesn't
        # support "SAME" and "VALID" padding, so padding must be explicitly specified.
        ((1, 48, 48, 8), 8, (3, 3), 8, (1, 1), 1),
        ((1, 48, 48, 16), 16, (3, 3), 16, (2, 2), (1, 1, 0, 0)),
        ((1, 24, 24, 32), 32, (3, 3), 32, (1, 1), 1),
        ((1, 24, 24, 32), 32, (3, 3), 32, (2, 2), (1, 1, 0, 0)),
        ((1, 12, 12, 64), 64, (3, 3), 64, (1, 1), 1),
        ((1, 12, 12, 64), 64, (3, 3), 64, (2, 2), (1, 1, 0, 0)),
        ((1, 6, 6, 128), 128, (3, 3), 128, (1, 1), 1),
        ((1, 6, 6, 128), 128, (3, 3), 128, (2, 2), (1, 1, 0, 0)),
        ((1, 3, 3, 256), 256, (3, 3), 256, (1, 1), 1),
        # Asymmetric height and width
        ((1, 25, 5, 64), 64, (3, 3), 64, (1, 1), 1),
        # Larger kernel
        ((1, 24, 24, 8), 8, (5, 5), 8, (1, 1), 1),
        # Asymmetric kernel
        ((1, 24, 24, 8), 8, (3, 5), 8, (1, 1), 1),
    ]

    data_shape, groups, kernel_size, num_filter, strides, padding, in_dtype = tvm.testing.parameters(
        # Make a copy of each parameterized test for int8 and one for int16
        *map(lambda t: t + ("int8",), in_dtype_parameterized_tests),
        *map(lambda t: t + ("int16",), in_dtype_parameterized_tests),
        # Test the int16 implementation with channel numbers not divisible by four
        ((1, 48, 48, 6), (3, 3), 6, (1, 1), 1, "int16"),
    )
    dilation = tvm.testing.parameter(1)
    data_layout, kernel_layout, out_layout = tvm.testing.parameters(("NHWC", "HWOI", "NHWC"),)
    schedule_name = tvm.testing.parameter("depthwise_conv2d_nhwc_dsp.arm_cpu")


class TestDepthwiseConv2d_Tensordot(GeneralizedConv2dTests):
    data_shape, groups, kernel_size, num_filter, strides, padding, in_dtype = tvm.testing.parameters(
        # Currently, our schedule requires kernel_w be divisible by the number of simd lanes given
        # its dtype. This means 3x3 and 5x5 kernels do not work on int16 or int8 for now. If you had
        # to, you could hack around this by padding the data and kernel.
        ((1, 8, 48, 48), 48, (3, 3), 8, (1, 1), 1, "int32"),
        ((1, 16, 48, 48), 48, (3, 3), 16, (2, 2), (1, 1, 0, 0), "int32"),
        ((1, 32, 24, 24), 24, (3, 3), 32, (1, 1), 1, "int32"),
        ((1, 32, 24, 24), 24, (3, 3), 32, (2, 2), (1, 1, 0, 0), "int32"),
        ((1, 64, 12, 12), 12, (3, 3), 64, (1, 1), 1, "int32"),
        ((1, 64, 12, 12), 12, (3, 3), 64, (2, 2), (1, 1, 0, 0), "int32"),
        ((1, 128, 6, 6), 6, (3, 3), 128, (1, 1), 1, "int32"),
        ((1, 128, 6, 6), 6, (3, 3), 128, (2, 2), (1, 1, 0, 0), "int32"),
        ((1, 256, 3, 3), 3, (3, 3), 256, (1, 1), 1, "int32"),
        ((1, 64, 25, 5), 5, (3, 3), 64, (1, 1), 1, "int32"),
        ((1, 8, 24, 24), 24, (5, 5), 8, (1, 1), 1, "int32"),
        ((1, 8, 24, 24), 24, (3, 5), 8, (1, 1), 1, "int32"),
        # These "evenly divisible" kernels work on smaller dtypes.
        ((1, 8, 48, 48), 48, (3, 2), 8, 1, 0, "int16"),
        ((1, 8, 48, 48), 48, (4, 4), 8, 1, 0, "int8"),
    )
    dilation = tvm.testing.parameter(1)
    data_layout = tvm.testing.parameters(("NCHW", "OIHW",))
    out_layout = tvm.testing.parameter("NHWC", "NCHW")
    schedule_name = tvm.testing.parameter("depthwise_conv2d_nchw_oihw_dsp.arm_cpu")


if __name__ == "__main__":
    tvm.testing.main()
