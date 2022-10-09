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

random_seed = tvm.testing.parameter(0)


@tvm.testing.fixture
def shift_shape(batch):
    return batch


@tvm.testing.fixture
def shift_shape(in_channel):
    return in_channel


@tvm.testing.fixture
def shift_shape(in_size):
    return in_size


@tvm.testing.fixture
def shift_shape(num_filter):
    return num_filter


@tvm.testing.fixture
def shift_shape(stride):
    return stride


@tvm.testing.fixture
def shift_shape(padding):
    return padding


@tvm.testing.fixture
def shift_shape(output_padding):
    return output_padding


class BaseConv2DTransposeTests:
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
        in_height, in_width = in_size
        kernel_height, kernel_width = (1, 1)
        stride_height, stride_width = stride
        pad_top, pad_left, pad_bottom, pad_right = padding

        A = te.placeholder((batch, in_channel, in_height, in_width), name="A")
        W = te.placeholder((in_channel, num_filter, kernel_height, kernel_width), name="W")

        a_shape = get_const_tuple(A.shape)
        w_shape = get_const_tuple(W.shape)
        dtype = A.dtype

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
            A,
            W,
            [stride_height, stride_width],
            [pad_top, pad_left, pad_bottom, pad_right],
            A.dtype,
            output_padding,
        )

        with tvm.target.Target(get_hexagon_target("v68")):
            fcompute = topi.nn.conv2d_transpose_nchw
            fschedule = topi.hexagon.schedule_conv2d_transpose_nchw
            B = fcompute(*fcompute_args)
            C = topi.nn.relu(B)
            s1 = fschedule([B])
            s2 = fschedule([C])

            dev = hexagon_session.device

            a = tvm.nd.array(a_np, dev)
            w = tvm.nd.array(w_np, dev)
            b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
            c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)

            func1 = tvm.build(s1, [A, W, B], get_hexagon_target("v68"))
            func2 = tvm.build(s2, [A, W, C], get_hexagon_target("v68"))

            mod1 = hexagon_session.load_module(func1)
            mod2 = hexagon_session.load_module(func2)

            mod1(a, w, b)
            mod2(a, w, c)
            tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)
            tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)


class TestConv2DTranspose(BaseConv2DTransposeTests):

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
