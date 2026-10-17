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

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


@T.function
def ptx_cp_async(A: T.Tensor((32, 128), "float16"), B: T.Tensor((32, 128), "float16")) -> None:
    T.func_attr({"global_symbol": "default_function", "tirx.noalias": True})
    T.launch_thread("blockIdx.x", 1)
    tx = T.launch_thread("threadIdx.x", 32)
    A_shared = T.alloc_tensor(ty_args=[T.Tensor([32, 128], "float16", scope="shared")])
    for i in range(16):
        T.evaluate(
            T.s_tir.cp_async_raw.legacy(
                A_shared.data,
                tx * 128 + 8 * i,
                A.data,
                tx * 128 + 8 * i,
                16,
                elem_dtype="float16",
                ty="float16",
            )
        )
    T.evaluate(T.ptx.cp.async_.commit_group())
    T.evaluate(T.ptx.cp.async_.wait_group(0))
    for i in range(128):
        B[tx, i] = A_shared[tx, i]


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(8), reason="need cuda compute >= 8.0")
def test_ptx_cp_async():
    f = ptx_cp_async

    mod = tvm_tirx.tirx.compile(f, target="cuda")
    A_np = np.random.rand(32, 128).astype("float16")
    B_np = np.zeros((32, 128)).astype("float16")

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        A_nd = tvm_tirx.runtime.tensor(A_np, device=dev)
        B_nd = tvm_tirx.runtime.tensor(B_np, device=dev)
        mod(A_nd, B_nd)
        tvm_tirx.testing.assert_allclose(B_nd.numpy(), A_np)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_cp_async_raw_dtype_round_trips():
    # The raw cp.async form emitted by InjectPTXAsyncCopy carries the element
    # dtype in Call.dtype and must survive a TVMScript print -> parse round-trip
    # via T.s_tir.cp_async_raw with ty=. Guards the regression
    # where the element dtype was dropped after the flat op was phased out.
    @T.function
    def f(A: T.Tensor((128,), "float16"), B: T.Tensor((128,), "float16")):
        T.func_attr({"global_symbol": "f"})
        for i in T.serial(8):
            T.s_tir.cp_async_raw(B.data, i * 16, A.data, i * 16, 16, ty="float16")

    reparsed = tvm_tirx.script.from_source(
        f.script(),
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx},
    )
    tvm_tirx.ir.assert_structural_equal(f, reparsed)


# Note: tests for the indexed barrier API (`create_barriers`,
# `ptx_init_barrier_thread_count`, `ptx_arrive_barrier`, `ptx_wait_barrier`,
# `ptx_cp_async_barrier`, `ptx_arrive_barrier_expect_tx`) were removed —
# fork uses `ptx_mbarrier_*` instead and those intrinsics have no
# users elsewhere in this codebase.


if __name__ == "__main__":
    test_ptx_cp_async()
