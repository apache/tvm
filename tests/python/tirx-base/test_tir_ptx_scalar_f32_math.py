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

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env


@T.prim_func(s_tir=True)
def ptx_scalar_f32_math(
    A: T.Buffer((32,), "float32"),
    B: T.Buffer((32,), "float32"),
    C_add: T.Buffer((32,), "float32"),
    C_mul: T.Buffer((32,), "float32"),
    C_max: T.Buffer((32,), "float32"),
) -> None:
    T.func_attr({"global_symbol": "default_function", "tirx.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.sblock():
        T.reads(A[0:32], B[0:32])
        T.writes(C_add[0:32], C_mul[0:32], C_max[0:32])
        T.evaluate(T.ptx.add_f32(T.address_of(C_add[tx]), A[tx], B[tx]))
        T.evaluate(T.ptx.mul_f32(T.address_of(C_mul[tx]), A[tx], B[tx]))
        C_max[tx] = T.ptx.max_f32(A[tx], B[tx])


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(7), reason="need cuda compute >= 7.0")
def test_ptx_scalar_f32_math():
    f = ptx_scalar_f32_math
    mod = tvm.compile(f, target="cuda")
    rng = np.random.default_rng(0)
    A_np = rng.standard_normal(32).astype("float32")
    B_np = rng.standard_normal(32).astype("float32")
    Z = np.zeros((32,), dtype="float32")

    def run_and_check():
        dev = tvm.cuda(0)
        A_nd = tvm.runtime.tensor(A_np, device=dev)
        B_nd = tvm.runtime.tensor(B_np, device=dev)
        Cadd = tvm.runtime.tensor(Z.copy(), device=dev)
        Cmul = tvm.runtime.tensor(Z.copy(), device=dev)
        Cmax = tvm.runtime.tensor(Z.copy(), device=dev)
        mod(A_nd, B_nd, Cadd, Cmul, Cmax)
        tvm.testing.assert_allclose(Cadd.numpy(), A_np + B_np, rtol=0, atol=0)
        tvm.testing.assert_allclose(Cmul.numpy(), A_np * B_np, rtol=0, atol=0)
        tvm.testing.assert_allclose(Cmax.numpy(), np.maximum(A_np, B_np), rtol=0, atol=0)

    tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    test_ptx_scalar_f32_math()
