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

class TestGroupConv2d_NCHW_OIHW(GeneralizedConv2dTests):
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
    in_dtype = tvm.testing.parameter("int8", "int16")
    data_layout, kernel_layout, out_layout = tvm.testing.parameters(("NCHW", "OIHW", "NCHW"),)
    schedule_name = tvm.testing.parameter("group_conv2d_nchw.arm_cpu")


class TestGroupConv2d_NHWC_HWIO(GeneralizedConv2dTests):
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
    in_dtype = tvm.testing.parameter("int8", "int16")
    data_layout, kernel_layout, out_layout = tvm.testing.parameters(("NHWC", "HWIO", "NHWC"),)
    schedule_name = tvm.testing.parameter("group_conv2d_nhwc.generic")


if __name__ == "__main__":
    tvm.testing.main()
