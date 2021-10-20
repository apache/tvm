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


def convert_int32_into_int4(a_int32):
    """convert int32 values into int4
    Parameters
    ----------
    a_int32 : int

    Return
    ------
    a_int4 : int
    """
    B, K, L = a_int32.shape
    assert L % 8 == 0
    a_int4 = np.zeros(shape=(B, K, L // 8), dtype=np.int32)
    for b in range(B):
        for k in range(K):
            for l in range(L // 8):
                for m in range(min(8, L - l * 8)):
                    a_int4[b, k, l] = a_int4[b, k, l] | (
                        (a_int32[b, k, l * 8 + m] & 0xF) << ((7 - m) * 4)
                    )
    return a_int4


def verify_batch_matmul(x_batch, y_batch, M, N, K, dtype):
    x = te.placeholder((x_batch, M, K), name="x", dtype=dtype)
    y = te.placeholder((y_batch, N, K), name="y", dtype=dtype)

    assert dtype in ["int4", "int8", "float16"]

    out_dtype = "float32"
    if dtype in ["int8", "int4"]:
        out_dtype = "int32"

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_batch_matmul_tensorcore")
    def get_ref_data():
        if dtype == "int4":
            a_np = np.random.randint(low=-8, high=7, size=(x_batch, M, K))
            b_np = np.random.randint(low=-8, high=7, size=(y_batch, N, K))
        elif dtype == "int8":
            a_np = np.random.randint(low=-128, high=127, size=(x_batch, M, K)).astype(dtype)
            b_np = np.random.randint(low=-128, high=127, size=(y_batch, N, K)).astype(dtype)
        else:
            a_np = np.random.uniform(size=(x_batch, M, K)).astype(dtype)
            b_np = np.random.uniform(size=(y_batch, N, K)).astype(dtype)
        c_np = tvm.topi.testing.batch_matmul(a_np, b_np, out_dtype)
        return (a_np, b_np, c_np)

    # get the test data
    a_np, b_np, c_np = get_ref_data()
    if dtype == "int4":
        a_np = convert_int32_into_int4(a_np)
        b_np = convert_int32_into_int4(b_np)

    def check_device(device):
        dev = tvm.device(device, 0)
        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            fcompute, fschedule = tvm.topi.testing.dispatch(device, _batch_matmul_implement)
            out = fcompute(x, y, None, out_dtype)
            s = fschedule([out])
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=out_dtype), dev)
        f = tvm.build(s, [x, y, out], device, name="batch_matmul")
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    check_device("cuda")


@tvm.testing.requires_tensorcore
def test_batch_matmul():
    for dtype in ["float16", "int8", "int4"]:
        verify_batch_matmul(1, 1, 16, 16, 32, dtype)
        verify_batch_matmul(5, 5, 16, 16, 32, dtype)
        verify_batch_matmul(5, 5, 16, 32, 32, dtype)
        verify_batch_matmul(30, 30, 16, 32, 32, dtype)


if __name__ == "__main__":
    test_batch_matmul()
