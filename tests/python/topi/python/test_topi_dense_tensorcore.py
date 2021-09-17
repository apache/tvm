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
from tvm.topi.utils import get_const_tuple
from tvm import te
from tvm.contrib.pickle_memoize import memoize
import tvm.testing


_dense_implement = {"gpu": [(topi.cuda.dense_tensorcore, topi.cuda.schedule_dense_tensorcore)]}


def convert_int32_into_int4(a_int32):
    """convert int32 values into int4
    Parameters
    ----------
    a_int32 : int

    Return
    ------
    a_int4 : int
    """
    K, L = a_int32.shape
    assert L % 8 == 0
    a_int4 = np.zeros(shape=(K, L // 8), dtype=np.int32)
    for k in range(K):
        for l in range(L // 8):
            for m in range(min(8, L - l * 8)):
                a_int4[k, l] = a_int4[k, l] | ((a_int32[k, l * 8 + m] & 0xF) << ((7 - m) * 4))
    return a_int4


def convert_int32_into_int4_bias(a_int32):
    """convert int32 values into int4
    Parameters
    ----------
    a_int32 : int

    Return
    ------
    a_int4 : int
    """
    (L,) = a_int32.shape
    assert L % 8 == 0
    a_int4 = np.zeros(shape=(L // 8), dtype=np.int32)
    for l in range(L // 8):
        for m in range(min(8, L - l * 8)):
            a_int4[l] = a_int4[l] | ((a_int32[l * 8 + m] & 0xF) << ((7 - m) * 4))
    return a_int4


def verify_dense(batch, in_dim, out_dim, dtype, use_bias=True):
    """Dense tensorcore verify function"""
    A = te.placeholder((batch, in_dim), name="A", dtype=dtype)
    B = te.placeholder((out_dim, in_dim), name="B", dtype=dtype)
    C = te.placeholder((out_dim,), name="C", dtype=dtype)

    assert dtype in ["int4", "int8", "float16"]

    out_dtype = "float32"
    if dtype in ["int8", "int4"]:
        out_dtype = "int32"

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_dense_tensorcore")
    def get_ref_data():
        if dtype == "int4":
            a_np = np.random.randint(low=-8, high=7, size=(batch, in_dim))
            b_np = np.random.randint(low=-8, high=7, size=(out_dim, in_dim))
            c_np = np.random.randint(low=-8, high=7, size=(out_dim,))
        elif dtype == "int8":
            a_np = np.random.randint(low=-128, high=127, size=(batch, in_dim)).astype(dtype)
            b_np = np.random.randint(low=-128, high=127, size=(out_dim, in_dim)).astype(dtype)
            c_np = np.random.randint(low=-128, high=127, size=(out_dim,)).astype(dtype)
        else:
            a_np = np.random.uniform(size=(batch, in_dim)).astype(dtype)
            b_np = np.random.uniform(size=(out_dim, in_dim)).astype(dtype)
            c_np = np.random.uniform(size=(out_dim,)).astype(dtype)
        d_np = tvm.topi.testing.dense(a_np, b_np, c_np, use_bias, True, out_dtype)
        return (a_np, b_np, c_np, d_np)

    # get the test data
    a_np, b_np, c_np, d_np = get_ref_data()
    if dtype == "int4":
        a_np = convert_int32_into_int4(a_np)
        b_np = convert_int32_into_int4(b_np)
        c_np = convert_int32_into_int4_bias(c_np)

    def check_device(device):
        dev = tvm.device(device, 0)
        print("Running on target: %s" % device)
        for fcompute, fschedule in tvm.topi.testing.dispatch(device, _dense_implement):
            with tvm.target.Target(device):
                D = fcompute(A, B, C if use_bias else None, out_dtype)
                D = topi.nn.relu(D)
                s = fschedule([D])
            a = tvm.nd.array(a_np, dev)
            b = tvm.nd.array(b_np, dev)
            c = tvm.nd.array(c_np, dev)
            d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=out_dtype), dev)
            f = tvm.build(s, [A, B, C, D], device, name="dense")
            f(a, b, c, d)
            tvm.testing.assert_allclose(d.numpy(), d_np, rtol=1e-3)

    check_device("cuda")


@tvm.testing.requires_tensorcore
def test_dense_tensorcore():
    """Test cases"""
    for dtype in ["float16", "int8"]:
        verify_dense(8, 16, 32, "float16", use_bias=True)
        verify_dense(16, 32, 16, dtype, use_bias=True)
        verify_dense(256, 1024, 1024, dtype, use_bias=True)
        verify_dense(1000, 1024, 1024, dtype, use_bias=False)
        verify_dense(256, 2048, 1000, dtype, use_bias=False)
    # TODO: need fix int4 use_bias=True, wyc-ruiker
    verify_dense(16, 32, 16, "int4", use_bias=False)
    verify_dense(256, 1024, 1024, "int4", use_bias=False)
    verify_dense(1000, 1024, 1024, "int4", use_bias=False)
    verify_dense(256, 2048, 1000, "int4", use_bias=False)


if __name__ == "__main__":
    test_dense_tensorcore()
