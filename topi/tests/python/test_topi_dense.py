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
"""Test code for dense operator"""
import numpy as np
import tvm
import topi
import topi.testing
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize

from common import get_all_backend, Int8Fallback

def verify_dense(batch, in_dim, out_dim, use_bias=True):
    A = tvm.placeholder((batch, in_dim), name='A')
    B = tvm.placeholder((out_dim, in_dim), name='B')
    C = tvm.placeholder((out_dim,), name='C')
    dtype = A.dtype

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_dense")
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, in_dim)).astype(dtype)
        b_np = np.random.uniform(size=(out_dim, in_dim)).astype(dtype)
        c_np = np.random.uniform(size=(out_dim,)).astype(dtype)
        if use_bias:
            d_np = np.maximum(np.dot(a_np, b_np.T) + c_np, 0.0)
        else:
            d_np = np.maximum(np.dot(a_np, b_np.T), 0.0)
        return (a_np, b_np, c_np, d_np)
    # get the test data
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            D = topi.nn.dense(A, B, C if use_bias else None)
            D = topi.nn.relu(D)
            s = topi.generic.schedule_dense([D])
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B, C, D], device, name="dense")
        f(a, b, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)


def verify_dense_int8(batch, in_dim, out_dim, use_bias=True):
    dtype = 'int8'
    out_dtype = 'int32'
    A = tvm.placeholder((batch, in_dim), name='A', dtype=dtype)
    B = tvm.placeholder((out_dim, in_dim), name='B', dtype=dtype)
    C = tvm.placeholder((out_dim,), name='C', dtype=out_dtype)

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_dense_int8")
    def get_ref_data():
        a_np = np.random.randint(low=-128, high=127, size=(batch, in_dim)).astype(dtype)
        b_np = np.random.randint(low=-128, high=127, size=(out_dim, in_dim)).astype(dtype)
        c_np = np.random.randint(low=-128, high=127, size=(out_dim,)).astype(out_dtype)
        d_np = np.dot(a_np.astype(out_dtype), b_np.T.astype(out_dtype))
        if use_bias:
            d_np += c_np
        d_np = np.maximum(d_np, 0.0)
        return (a_np, b_np, c_np, d_np)

    # get the test data
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        if device == "cuda" and not tvm.contrib.nvcc.have_int8(ctx.compute_version):
            print("Skip because int8 intrinsics are not available")
            return

        print("Running on target: %s" % device)
        with tvm.target.create(device):
            D = topi.nn.dense(A, B, C if use_bias else None, out_dtype=out_dtype)
            D = topi.nn.relu(D)
            s = topi.generic.schedule_dense([D])
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=out_dtype), ctx)
        f = tvm.build(s, [A, B, C, D], device, name="dense")
        f(a, b, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-5)

    for device in ['cuda']:
        check_device(device)


def test_dense():
    verify_dense(1, 1024, 1000, use_bias=True)
    verify_dense(1, 1024, 1000, use_bias=False)

    verify_dense(2, 1024, 1000, use_bias=True)


def test_dense_int8():
    with Int8Fallback():
        verify_dense_int8(2, 1024, 1000, use_bias=True)
        verify_dense_int8(2, 1024, 1000, use_bias=False)


if __name__ == "__main__":
    test_dense()
    test_dense_int8()
