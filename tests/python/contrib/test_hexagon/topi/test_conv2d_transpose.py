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
"""Test code for transposed convolution."""
import numpy as np

import tvm
from tvm.contrib.hexagon.session import Session
import tvm.testing
from tvm import te
from tvm import topi
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple

from ..infrastructure import get_hexagon_target

# TODO Should add kernal to tvm.testing.fixture


class BaseConv2DTransposeTests:
    """Conv2D transpose base class."""

    random_seed = tvm.testing.parameter(0)

    @tvm.testing.requires_hexagon
    def test_conv2d(
        self,
        hexagon_session: Session,
        batch,
        in_channel,
        in_size,
        num_filter,
        stride,
        padding,
        output_padding,
        random_seed,
    ):
        """Test conv2D."""
        in_height, in_width = in_size
        kernel_height, kernel_width = (1, 1)
        stride_height, stride_width = stride
        pad_top, pad_left, pad_bottom, pad_right = padding

        a_tensor = te.placeholder((batch, in_channel, in_height, in_width), name="a_tensor")
        w_tensor = te.placeholder(
            (in_channel, num_filter, kernel_height, kernel_width), name="w_tensor"
        )

        a_shape = get_const_tuple(a_tensor.shape)
        w_shape = get_const_tuple(w_tensor.shape)
        dtype = a_tensor.dtype

        def get_ref_data():

            np.random.seed(random_seed)
            a_np = np.random.uniform(size=a_shape).astype(dtype)
            w_np = np.random.uniform(size=w_shape).astype(dtype)
            b_np = tvm.topi.testing.conv2d_transpose_nchw_python(
                a_np, w_np, stride, padding, output_padding
            )
            c_np = np.maximum(b_np, 0)
            return a_np, w_np, b_np, c_np

        a_np, w_np, b_np, c_np = get_ref_data()

        fcompute_args = (
            a_tensor,
            w_tensor,
            [stride_height, stride_width],
            [pad_top, pad_left, pad_bottom, pad_right],
            a_tensor.dtype,
            output_padding,
        )

        with tvm.target.Target(get_hexagon_target("v68")):
            fcompute = topi.nn.conv2d_transpose_nchw
            fschedule = topi.hexagon.schedule_conv2d_transpose_nchw
            b_tensor = fcompute(*fcompute_args)
            c_tensor = topi.nn.relu(b_tensor)
            schedule_1 = fschedule([b_tensor])
            schedule_2 = fschedule([c_tensor])

            dev = hexagon_session.device

            a_data = tvm.nd.array(a_np, dev)
            weight = tvm.nd.array(w_np, dev)
            b = tvm.nd.array(np.zeros(get_const_tuple(b_tensor.shape), dtype=b_tensor.dtype), dev)
            c = tvm.nd.array(np.zeros(get_const_tuple(c_tensor.shape), dtype=c_tensor.dtype), dev)

            func1 = tvm.build(schedule_1, [a_tensor, w_tensor, b_tensor], get_hexagon_target("v68"))
            func2 = tvm.build(schedule_2, [a_tensor, w_tensor, c_tensor], get_hexagon_target("v68"))

            mod1 = hexagon_session.load_module(func1)
            mod2 = hexagon_session.load_module(func2)

            mod1(a_data, weight, b)
            mod2(a_data, weight, c)
            tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)
            tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)


class TestConv2DTranspose(BaseConv2DTransposeTests):
    """Test Conv2D transpose class."""

    (batch, in_channel, in_size, num_filter, stride) = tvm.testing.parameters(
        (1, 3, (224, 224), 1, (1, 1)),
        (1, 8, (224, 224), 1, (1, 1)),
        (1, 512, (8, 1), 128, (31, 1)),
        (1, 32, (8192, 1), 1, (1, 1)),
    )

    padding = tvm.testing.parameter((0, 0, 0, 0))
    output_padding = tvm.testing.parameter((0, 0))


if __name__ == "__main__":
    tvm.testing.main()
