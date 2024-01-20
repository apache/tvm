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
"""Test code for group transposed 1d convolution."""

import itertools
import os

import numpy as np

import tvm
import tvm.testing
import tvm.topi.testing

from tvm import te, topi
from tvm.topi.utils import get_const_tuple

_group_conv1d_transpose_ncw_implement = {
    "generic": (
        topi.nn.group_conv1d_transpose_ncw,
        topi.generic.schedule_group_conv1d_transpose_ncw,
    ),
}


(
    batch,
    in_channel,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    output_padding,
    groups,
) = tvm.testing.parameters(
    (1, 4, 224, 32, 5, 1, 0, (0,), 4),
    (1, 8, 224, 32, 7, 1, 2, (0,), 4),
    (1, 8, 224, 32, 5, 2, 1, (0,), 2),
    (1, 4, 224, 4, 5, 2, 1, (1,), 4),
    (1, 3, 224, 15, 5, 2, 0, (0,), 3),
    (1, 32, 32, 128, 5, 1, 0, (0,), 32),
    (1, 32, 32, 128, 5, 2, 1, (0,), 16),
)

dtype = tvm.testing.parameter("float32")


@tvm.testing.fixture(cache_return_value=True)
def ref_data(
    dtype, batch, in_channel, in_size, num_filter, kernel, stride, padding, output_padding, groups
):
    dtype = "float32"
    a_shape = (batch, in_channel, in_size)
    w_shape = (in_channel, num_filter, kernel)

    a_np = np.random.uniform(size=a_shape).astype(dtype)
    w_np = np.random.uniform(size=w_shape).astype(dtype)
    b_np = tvm.topi.testing.group_conv1d_transpose_ncw_python(
        a_np, w_np, stride, padding, output_padding, groups
    )
    c_np = np.maximum(b_np, 0)
    return a_np, w_np, b_np, c_np


@tvm.testing.known_failing_targets("cuda", "vulkan")
def test_group_conv1d_transpose_ncw(
    target, dev, ref_data, dtype, stride, padding, output_padding, groups
):
    a_np, w_np, b_np, c_np = ref_data

    A = te.placeholder(a_np.shape, name="A", dtype=dtype)
    W = te.placeholder(w_np.shape, name="W", dtype=dtype)

    with tvm.target.Target(target):
        fcompute, fschedule = tvm.topi.testing.dispatch(
            target, _group_conv1d_transpose_ncw_implement
        )
        B = fcompute(A, W, stride, padding, A.dtype, output_padding, groups)
        C = topi.nn.relu(B)
        s1 = fschedule([B])
        s2 = fschedule([C])
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)

    func1 = tvm.build(s1, [A, W, B], target)
    func2 = tvm.build(s2, [A, W, C], target)
    func1(a, w, b)
    func2(a, w, c)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
