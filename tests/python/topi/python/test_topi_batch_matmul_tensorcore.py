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
"""Test code for batch_matmul operator"""
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple
from tvm.contrib.pickle_memoize import memoize

import tvm.testing

_batch_matmul_implement = {
    "gpu": (topi.cuda.batch_matmul_tensorcore, topi.cuda.schedule_batch_matmul_tensorcore),
}


def verify_batch_matmul(x_batch, y_batch, M, N, K):
    x = te.placeholder((x_batch, M, K), name="x")
    y = te.placeholder((y_batch, N, K), name="y")
    dtype = x.dtype

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_batch_matmul_tensorcore")
    def get_ref_data():
        a_np = np.random.uniform(size=(x_batch, M, K)).astype(dtype)
        b_np = np.random.uniform(size=(y_batch, N, K)).astype(dtype)
        c_np = tvm.topi.testing.batch_matmul(a_np, b_np)
        return (a_np, b_np, c_np)

    # get the test data
    a_np, b_np, c_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            fcompute, fschedule = tvm.topi.testing.dispatch(device, _batch_matmul_implement)
            out = fcompute(x, y)
            s = fschedule([out])
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=dtype), ctx)
        f = tvm.build(s, [x, y, out], device, name="dense")
        f(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-3)

    check_device("cuda")


@tvm.testing.uses_gpu
def test_batch_matmul():
    verify_batch_matmul(1, 1, 16, 16, 32)
    verify_batch_matmul(5, 5, 16, 16, 32)
    verify_batch_matmul(5, 5, 16, 32, 32)
    verify_batch_matmul(30, 30, 16, 32, 32)


if __name__ == "__main__":
    test_batch_matmul()
