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
import contextlib
import numpy as np
import pytest
import sys

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import te, topi
from tvm.topi.utils import get_const_tuple

from common import Int8Fallback

random_seed = tvm.testing.parameter(0)

use_bias = tvm.testing.parameter(True, False)
batch_size = tvm.testing.parameter(1, 2, 128)
in_dim, out_dim = tvm.testing.parameters((1024, 1000))
in_dtype, out_dtype = tvm.testing.parameters(
    ("float32", "float32"),
    ("float16", "float16"),
    ("int8", "int32"),
)


_dense_implementations = {
    "generic": [(topi.nn.dense, topi.generic.schedule_dense)],
    "cpu": [
        (topi.x86.dense_nopack, topi.x86.schedule_dense_nopack),
        (topi.x86.dense_pack, topi.x86.schedule_dense_pack),
        (topi.x86.dense_dynamic, topi.x86.schedule_dense_dynamic),
    ],
    "gpu": [
        (topi.gpu.dense_small_batch, topi.gpu.schedule_dense_small_batch),
        (topi.gpu.dense_large_batch, topi.gpu.schedule_dense_large_batch),
    ],
    "mali": [(topi.mali.dense, topi.mali.schedule_dense)],
    "bifrost": [(topi.bifrost.dense, topi.bifrost.schedule_dense)],
    "hls": [(topi.nn.dense, topi.hls.schedule_dense)],
}


@tvm.testing.fixture(cache_return_value=True)
def dense_ref_data(random_seed, batch_size, in_dim, out_dim, use_bias, in_dtype, out_dtype):
    np.random.seed(random_seed)

    if "float" in in_dtype:
        a_np = np.random.uniform(size=(batch_size, in_dim)).astype(in_dtype)
        b_np = np.random.uniform(size=(out_dim, in_dim)).astype(in_dtype)
        c_np = np.random.uniform(size=(out_dim,)).astype(out_dtype)
    elif in_dtype == "int8":
        a_np = np.random.randint(low=-128, high=127, size=(batch_size, in_dim)).astype(in_dtype)
        b_np = np.random.randint(low=-128, high=127, size=(out_dim, in_dim)).astype(in_dtype)
        c_np = np.random.randint(low=-128, high=127, size=(out_dim,)).astype(out_dtype)
    else:
        raise ValueError("No method to generate test data for data type '{}'".format(in_dtype))

    matmul = np.dot(a_np.astype(out_dtype), b_np.T.astype(out_dtype))

    if use_bias:
        matmul += c_np

    d_np = np.maximum(matmul, 0)
    return (a_np, b_np, c_np, d_np)


def test_dense(
    target,
    dev,
    batch_size,
    in_dim,
    out_dim,
    use_bias,
    dense_ref_data,
    in_dtype,
    out_dtype,
    implementations=None,
):
    target = tvm.target.Target(target)

    if target.kind.name == "cuda":
        if in_dtype == "int8" and not tvm.contrib.nvcc.have_int8(dev.compute_version):
            pytest.xfail("CUDA int8 intrinsics not available")

        if in_dtype == "float16" and not tvm.contrib.nvcc.have_fp16(dev.compute_version):
            pytest.xfail("CUDA float16 intrinsics not available")

    if target.kind.name == "vulkan":
        if in_dtype == "int8" and (
            not target.attrs.get("supports_int8", False)
            or not target.attrs.get("supports_8bit_buffer", False)
        ):
            pytest.xfail("Vulkan int8 driver support not available")
        if in_dtype == "float16" and (
            not target.attrs.get("supports_float16", False)
            or not target.attrs.get("supports_16bit_buffer", False)
        ):
            pytest.xfail("Vulkan float16 driver support not available")

    if (
        target.kind.name not in ["llvm", "c"]
        and len(set(target.keys) & set(_dense_implementations)) == 0
    ):
        pytest.xfail("No implementation for tvm.topi.testing.dispatch to find")

    if "int" in in_dtype:
        tol = {"atol": 0, "rtol": 0}
    elif in_dtype == "float32":
        tol = {"rtol": 1e-5, "atol": 1e-5}
    elif in_dtype == "float16":
        tol = {"rtol": 5e-2, "atol": 1e-5}

    A = te.placeholder((batch_size, in_dim), name="A", dtype=in_dtype)
    B = te.placeholder((out_dim, in_dim), name="B", dtype=in_dtype)
    C = te.placeholder((out_dim,), name="C", dtype=out_dtype)

    a_np, b_np, c_np, d_np = dense_ref_data

    if implementations is None:
        implementations = tvm.topi.testing.dispatch(target, _dense_implementations)

    for fcompute, fschedule in implementations:
        if fcompute == topi.x86.dense_dynamic and (batch_size != 1 or in_dtype != "float32"):
            continue
        with tvm.target.Target(target):
            D = fcompute(A, B, C if use_bias else None, out_dtype)
            D = topi.nn.relu(D)
            s = fschedule([D])

        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(c_np, dev)
        d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=out_dtype), dev)
        f = tvm.build(s, [A, B, C, D], target, name="dense")
        f(a, b, c, d)
        tvm.testing.assert_allclose(d.numpy(), d_np, **tol)


@pytest.mark.parametrize("target,in_dtype,out_dtype", [("cuda", "int8", "int32")])
def test_dense_cuda_int8(
    target,
    dev,
    batch_size,
    in_dim,
    out_dim,
    use_bias,
    dense_ref_data,
    in_dtype,
    out_dtype,
):
    implementations = [
        (topi.cuda.dense_int8, topi.cuda.schedule_dense_int8),
    ]
    with Int8Fallback():
        test_dense(
            target,
            dev,
            batch_size,
            in_dim,
            out_dim,
            use_bias,
            dense_ref_data,
            in_dtype,
            out_dtype,
            implementations=implementations,
        )


if __name__ == "__main__":
    tvm.testing.main()
