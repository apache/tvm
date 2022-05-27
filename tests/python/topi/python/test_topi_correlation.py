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
# under the License
"""test of correlation operator in NCHW layout"""
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing

from tvm import autotvm, te, topi

_correlation_implement = {
    "generic": (topi.nn.correlation_nchw, topi.generic.schedule_correlation_nchw),
    "gpu": (topi.cuda.correlation_nchw, topi.cuda.schedule_correlation_nchw),
}

(
    data_shape,
    kernel_size,
    max_displacement,
    stride1,
    stride2,
    pad_size,
    is_multiply,
) = tvm.testing.parameters(
    ((1, 3, 10, 10), 1, 4, 1, 1, 4, True),
    ((1, 3, 10, 10), 1, 5, 1, 1, 5, True),
    ((5, 1, 4, 4), 3, 1, 2, 1, 2, True),
    ((5, 1, 6, 4), 3, 1, 2, 2, 2, False),
    ((5, 1, 11, 11), 5, 1, 1, 1, 2, False),
)

dtype = tvm.testing.parameter("float32")


@tvm.testing.fixture(cache_return_value=True)
def ref_data(
    dtype, data_shape, kernel_size, max_displacement, stride1, stride2, pad_size, is_multiply
):
    a_np = np.random.uniform(size=data_shape).astype(dtype)
    b_np = np.random.uniform(size=data_shape).astype(dtype)
    c_np = tvm.topi.testing.correlation_nchw_python(
        a_np, b_np, kernel_size, max_displacement, stride1, stride2, pad_size, is_multiply
    )
    return a_np, b_np, c_np


def test_correlation_nchw(
    target,
    dev,
    ref_data,
    dtype,
    kernel_size,
    max_displacement,
    stride1,
    stride2,
    pad_size,
    is_multiply,
):
    a_np, b_np, c_np = ref_data

    A = te.placeholder(a_np.shape, name="data1", dtype=dtype)
    B = te.placeholder(b_np.shape, name="data2", dtype=dtype)

    fcompute, fschedule = tvm.topi.testing.dispatch(target, _correlation_implement)
    with tvm.target.Target(target):
        C = fcompute(A, B, kernel_size, max_displacement, stride1, stride2, pad_size, is_multiply)
        s = fschedule([C])

        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.empty(c_np.shape, dtype=dtype, device=dev)

        func = tvm.build(s, [A, B, C], target)
        func(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
