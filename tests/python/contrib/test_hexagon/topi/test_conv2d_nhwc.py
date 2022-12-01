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
"""Test code for convolution."""
import numpy as np

import tvm
import tvm.testing
from tvm import topi
from tvm import te
from tvm.contrib.hexagon.session import Session
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple

from ..infrastructure import get_hexagon_target


class BaseConv2DTests:
    """Test Conv2D base class."""

    dtype = tvm.testing.parameter("float32")

    @tvm.testing.fixture(cache_return_value=True)
    def ref_data(
        self, dtype, batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation
    ):
        """Generate reference data."""
        in_height = in_width = in_size
        a_shape = (batch, in_height, in_width, in_channel)
        w_shape = (kernel, kernel, in_channel, num_filter)

        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
        b_np = tvm.topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
        return a_np, w_np, b_np

    @tvm.testing.requires_hexagon
    def test_conv2d_nhwc(
        self,
        hexagon_session: Session,
        ref_data,
        batch,
        in_channel,
        in_size,
        num_filter,
        kernel,
        dtype,
        stride,
        padding,
        dilation,
    ):
        """Test Conv2D NHWC."""
        a_np, w_np, b_np = ref_data

        a_tensor = te.placeholder(a_np.shape, name="a_tensor", dtype=dtype)
        w_tensor = te.placeholder(w_np.shape, name="w_tensor", dtype=dtype)

        with tvm.target.Target(get_hexagon_target("v68")):
            fcompute = topi.nn.conv2d_nhwc
            fschedule = topi.hexagon.schedule_conv2d_nhwc
            b_tensor = fcompute(a_tensor, w_tensor, stride, padding, dilation, dtype)
            s = fschedule([b_tensor])

        func_name = "conv2d_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            dtype,
            batch,
            in_channel,
            in_size,
            num_filter,
            kernel,
            stride,
            padding,
            dilation,
        )
        func = tvm.build(
            s, [a_tensor, w_tensor, b_tensor], get_hexagon_target("v68"), name=func_name
        )
        mod = hexagon_session.load_module(func)

        dev = hexagon_session.device
        a_data = tvm.nd.array(a_np, dev)
        weight = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(np.zeros(get_const_tuple(b_tensor.shape), dtype=b_tensor.dtype), dev)

        mod[func_name](a_data, weight, b)
        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)


class TestConv2dNHWC(BaseConv2DTests):
    (
        batch,
        in_channel,
        in_size,
        num_filter,
        kernel,
        stride,
        padding,
        dilation,
    ) = tvm.testing.parameters(
        (1, 64, 32, 64, 3, 1, "SAME", 1),
        (4, 32, 16, 32, 5, 2, "SAME", 1),
        (1, 64, 32, 64, 3, 1, "VALID", 1),
        (4, 32, 16, 32, 5, 2, "VALID", 1),
        (1, 32, 16, 64, 3, 2, (0, 0, 1, 1), 1),
        (1, 32, 16, 64, 3, 2, (1, 1, 2, 2), 1),
        (1, 32, 16, 32, 5, 2, (3, 3, 2, 2), 1),
        (1, 32, 16, 64, 3, 2, (0, 1, 2, 3), 1),
        (1, 64, 32, 64, 3, 1, "SAME", 2),
        (1, 64, 32, 64, 3, 1, (1, 1, 2, 2), 2),
    )


if __name__ == "__main__":
    tvm.testing.main()
