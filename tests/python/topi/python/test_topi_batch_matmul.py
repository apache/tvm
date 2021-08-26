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
from common import Int8Fallback

_batch_matmul_implement = {
    "generic": (topi.nn.batch_matmul, topi.generic.schedule_batch_matmul),
    "cpu": (topi.x86.batch_matmul, topi.x86.schedule_batch_matmul),
    "gpu": (topi.cuda.batch_matmul, topi.cuda.schedule_batch_matmul),
}


def verify_batch_matmul(x_batch, y_batch, M, N, K, dynamic=False, debug=False):

    if not dynamic:
        x = te.placeholder((x_batch, M, K), name="x")
        y = te.placeholder((y_batch, N, K), name="y")
        dtype = x.dtype
    else:
        assert x_batch == y_batch or x_batch == 1 or y_batch == 1
        batch_size = max(x_batch, y_batch)
        dynamic_batch_size = te.var("dynamic_batch_size")
        dynamic_M = te.var("dynamic_M")
        dynamic_N = te.var("dynamic_N")
        dynamic_K = te.var("dynamic_K")

        x = te.placeholder((dynamic_batch_size, dynamic_M, dynamic_K), name="x")
        y = te.placeholder((dynamic_batch_size, dynamic_N, dynamic_K), name="y")
        dtype = x.dtype

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_batch_matmul")
    def get_ref_data():
        a_np = np.random.uniform(size=(x_batch, M, K)).astype(dtype)
        b_np = np.random.uniform(size=(y_batch, N, K)).astype(dtype)
        c_np = tvm.topi.testing.batch_matmul(a_np, b_np)
        return (a_np, b_np, c_np)

    # get the test data
    a_np, b_np, c_np = get_ref_data()

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            fcompute, fschedule = tvm.topi.testing.dispatch(target, _batch_matmul_implement)
            out = fcompute(x, y)
            if not dynamic:
                s = fschedule([out])
                out_shape = out.shape
            else:
                s = te.create_schedule(out.op)
                out_shape = (batch_size, M, N)

            if debug:
                print(tvm.lower(s, [x, y, out], simple_mode=True))

        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(out_shape), dtype=dtype), dev)
        f = tvm.build(s, [x, y, out], target, name="dense")
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    for target, dev in tvm.testing.enabled_targets():
        target_kind = tvm.target.Target(target).kind.name
        if dynamic and target_kind in ["cuda", "nvptx", "vulkan", "opencl"]:
            print("Dynamic batch matmul test is skippped on %s" % target)
            continue

        check_device(target, dev)


def verify_batch_matmul_int8(x_batch, y_batch, M, N, K):
    dtype = "int8"
    out_dtype = "int32"
    assert x_batch == y_batch or x_batch == 1 or y_batch == 1
    x = te.placeholder((x_batch, M, K), name="x", dtype=dtype)
    y = te.placeholder((y_batch, N, K), name="y", dtype=dtype)

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_batch_matmul")
    def get_ref_data():
        a_np = np.random.randint(low=-128, high=127, size=(x_batch, M, K)).astype(dtype)
        b_np = np.random.randint(low=-128, high=127, size=(y_batch, N, K)).astype(dtype)
        c_np = tvm.topi.testing.batch_matmul(a_np, b_np, out_dtype=out_dtype)
        return (a_np, b_np, c_np)

    # get the test data
    a_np, b_np, c_np = get_ref_data()

    def check_device(device):
        dev = tvm.device(device, 0)
        if device == "cuda" and not tvm.contrib.nvcc.have_int8(dev.compute_version):
            print("Skip because int8 intrinsics are not available")
            return

        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            out = topi.cuda.batch_matmul_int8(x, y, None, out_dtype)
            s = topi.cuda.schedule_batch_matmul_int8([out])
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=out_dtype), dev)
        f = tvm.build(s, [x, y, out], device, name="batch_matmul_int8")
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    for device in ["cuda"]:
        check_device(device)


@tvm.testing.uses_gpu
def test_batch_matmul():
    verify_batch_matmul(1, 1, 16, 16, 32)
    verify_batch_matmul(5, 5, 16, 16, 32)
    verify_batch_matmul(5, 5, 16, 20, 32)
    verify_batch_matmul(30, 30, 16, 20, 32)
    # Test batch broadcasting.
    verify_batch_matmul(1, 5, 16, 16, 32)
    verify_batch_matmul(5, 1, 16, 16, 32)

    # Test dynamic batch
    verify_batch_matmul(1, 1, 16, 16, 32, dynamic=True)
    verify_batch_matmul(5, 5, 16, 16, 32, dynamic=True)


@tvm.testing.requires_cuda
@tvm.testing.requires_gpu
def test_batch_matmul_int8():
    with Int8Fallback():
        verify_batch_matmul_int8(1, 1, 2, 3, 1)
        verify_batch_matmul_int8(1, 1, 16, 24, 32)
        verify_batch_matmul_int8(5, 5, 24, 16, 32)
        verify_batch_matmul_int8(30, 30, 16, 20, 32)
        verify_batch_matmul_int8(1, 5, 16, 16, 32)
        verify_batch_matmul_int8(5, 1, 16, 16, 32)


if __name__ == "__main__":
    test_batch_matmul()
    test_batch_matmul_int8()
