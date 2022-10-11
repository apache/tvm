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

class TestConv2d_NHWC_DSP(GeneralizedConv2dTests):
    """This test is for conv2d_nhwc_dsp.arm_cpu schedule."""

    groups = tvm.testing.parameter(1)
    data_shape, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        # TODO(mehrdadh): Fails due to https://github.com/apache/tvm/issues/11216
        # ((1, 32, 32, 1), (3, 3), 12, 1, 0, 1),
        # ((1, 32, 10, 3), (3, 3), 16, 1, 0, 1),
        # ((1, 49, 10, 1), (10, 4), 64, (2, 1), (4, 1, 5, 1), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 2),
        ((1, 32, 32, 16), (3, 3), 16, 1, (1, 1, 2, 2), 2),
        # from Keyword Spotting model from MLPerfTiny models
        # TODO(mehrdad): Fails due to https://github.com/apache/tvm/issues/11216
        # ((1, 49, 10, 1), (10, 4), 64, (2, 2), (4, 1, 5, 1), 1),
        # from Visual Wake Word model from MLPerfTiny models
        # TODO(mehrdadh): fails due to https://github.com/apache/tvm/issues/11216
        # ((1, 96, 96, 3), (3, 3), 8, (2, 2), (0, 0, 1, 1), 1),
        # from Image Classification model from MLPerfTiny models
        ((1, 16, 16, 32), (1, 1), 64, (2, 2), 0, 1),
        ((4, 16, 16, 8), (5, 5), 8, 2, (0, 4, 4, 0), 1),
        ((4, 16, 16, 8), (5, 5), 16, 2, (0, 4, 4, 0), 1),
        ((4, 16, 16, 8), (5, 5), 8, 2, 0, 1),
        ((4, 16, 16, 8), (5, 5), 16, 2, 0, 1),
        ((1, 16, 16, 8), (3, 3), 16, 2, (0, 0, 1, 1), 1),
        ((1, 16, 16, 8), (3, 3), 16, 2, (1, 1, 2, 2), 1),
        ((1, 16, 16, 8), (5, 5), 16, 2, (3, 3, 2, 2), 1),
        ((1, 16, 16, 8), (3, 3), 16, 2, (0, 1, 2, 3), 1),
    )
    in_dtype = tvm.testing.parameter("int8", "int16")
    data_layout, kernel_layout, out_layout = tvm.testing.parameters(("NHWC", "HWOI", "NHWC"),)
    schedule_name = tvm.testing.parameter("conv2d_nhwc_dsp.arm_cpu")


class TestConv2d_NHWC_Spatial_Pack(GeneralizedConv2dTests):
    """This test is for conv2d_nhwc_spatial_pack.arm_cpu schedule."""

    groups = tvm.testing.parameter(1)
    data_shape, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        ((1, 32, 32, 1), (3, 3), 12, 1, 0, 1),
        ((1, 32, 10, 3), (3, 3), 16, 1, 0, 1),
        ((1, 49, 10, 1), (10, 4), 64, (2, 1), (4, 1, 5, 1), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 2),
        ((1, 32, 32, 16), (3, 3), 16, 1, (1, 1, 2, 2), 2),
    )
    in_dtype = tvm.testing.parameter("int8", "int16")
    data_layout, kernel_layout, out_layout = tvm.testing.parameters(("NHWC", "HWIO", "NHWC"),)
    schedule_name = tvm.testing.parameter("conv2d_nhwc_spatial_pack.arm_cpu")


class TestConv2d_Tensordot(GeneralizedConv2dTests):

    groups = tvm.testing.parameter(1)
    data_shape, kernel_size, num_filter, strides, padding = tvm.testing.parameters(
        # Disabled because these kernels are not an integral number of words
        # ((1, 32, 32, 1), (3, 3), 12, 1, 0),
        # ((1, 32, 10, 3), (3, 3), 16, 1, 0),
        # ((1, 96, 96, 3), (3, 3), 8, (2, 2), (0, 0, 1, 1)),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0)),
        ((1, 16, 16, 32), (1, 1), 64, (2, 2), 0),
        ((1, 49, 10, 1), (10, 4), 64, (2, 1), (4, 1, 5, 1)),
        ((4, 16, 16, 16), (5, 5), 8, 2, 0),
    )
    dilation = tvm.testing.parameter(1)
    in_dtype = tvm.testing.parameter("int8", "int16", "int32")
    data_layout, kernel_layout = tvm.testing.parameters(("NHWC", "OHWI"),)
    out_layout = tvm.testing.parameter("NHWC", "NCHW")
    schedule_name = tvm.testing.parameter("conv2d_nhwc_ohwi_dsp.arm_cpu")


class TestConv2d_NCHW_Spatial_Pack(GeneralizedConv2dTests):
    """This test is for conv2d_nchw_spatial_pack.arm_cpu schedule."""

    groups = tvm.testing.parameter(1)
    data_shape, kernel_size, num_filter, strides, padding, dilation, in_dtype = tvm.testing.parameters(
        ((1, 32, 32, 16), (3, 3), 12, 1, 0, 1, "int8"),
        ((1, 32, 32, 16), (3, 3), 12, 1, 0, 1, "int16"),
        ((1, 16, 16, 32), (3, 3), 12, 1, 0, 1, "int16"),
    )
    data_layout, kernel_layout, out_layout = tvm.testing.parameters(("NCHW", "OIHW", "NCHW"),)
    schedule_name = tvm.testing.parameter("conv2d_nchw_spatial_pack.arm_cpu")



if __name__ == "__main__":
    tvm.testing.main()
