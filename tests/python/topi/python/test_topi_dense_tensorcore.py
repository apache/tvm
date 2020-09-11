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
# pylint: disable=invalid-name, too-many-locals, too-many-statements, unused-argument
"""Test code for dense tensorcore operator"""
import numpy as np
import tvm
from tvm import topi
import tvm.topi.testing
from tvm.topi.util import get_const_tuple
from tvm import te
from tvm.contrib.pickle_memoize import memoize
import tvm.testing


_dense_implement = {"gpu": [(topi.cuda.dense_tensorcore, topi.cuda.schedule_dense_tensorcore)]}


def verify_dense(batch, in_dim, out_dim, use_bias=True):
    """Dense tensorcore verify function"""
    A = te.placeholder((batch, in_dim), name="A")
    B = te.placeholder((out_dim, in_dim), name="B")
    C = te.placeholder((out_dim,), name="C")
    dtype = A.dtype

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_dense_tensorcore")
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
        print("Running on target: %s" % device)
        for fcompute, fschedule in tvm.topi.testing.dispatch(device, _dense_implement):
            with tvm.target.Target(device):
                D = fcompute(A, B, C if use_bias else None)
                D = topi.nn.relu(D)
                s = fschedule([D])
            a = tvm.nd.array(a_np, ctx)
            b = tvm.nd.array(b_np, ctx)
            c = tvm.nd.array(c_np, ctx)
            d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=dtype), ctx)
            f = tvm.build(s, [A, B, C, D], device, name="dense")
            f(a, b, c, d)
            tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-3)

    check_device("cuda")


@tvm.testing.requires_tensorcore
def test_dense_tensorcore():
    """Test cases"""
    verify_dense(8, 16, 32, use_bias=True)
    verify_dense(16, 32, 16, use_bias=True)
    verify_dense(256, 1024, 1024, use_bias=True)
    verify_dense(1000, 1024, 1024, use_bias=False)
    verify_dense(256, 2048, 1000, use_bias=False)


if __name__ == "__main__":
    test_dense_tensorcore()
