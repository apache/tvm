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

import pytest

import tvm
from tvm import tir
from tvm.script import tir as T


@T.prim_func
def __tir_kernel_exceed_smem__(A: T.handle) -> None:
    T.func_attr({"global_symbol": "__tir_kernel_exceed_smem__", "tir.noalias": True})
    A_buf = T.match_buffer(A, (1,), dtype="float32")
    # Create a trivial kernel environment
    for bx in T.thread_binding(1, thread="blockIdx.x"):
        for tx in T.thread_binding(1, thread="threadIdx.x"):
            # Intentionally allocate a large shared buffer to exceed typical 48KB caps
            # 64 * 1024 bytes using uint8 ensures deterministic size
            sh = T.alloc_buffer((64 * 1024,), dtype="uint8", scope="shared")
            # Dummy use to keep buffer live
            with T.block("use"):
                vi = T.axis.remap("S", [0])
                A_buf[0] = T.Cast("float32", sh[0])


def test_verify_gpu_code_flags_shared_mem_overflow():
    # Build an IRModule with the kernel
    mod = tvm.IRModule({"main": __tir_kernel_exceed_smem__})

    # Apply the verifier with a conservative cap (48KB)
    cap = 48 * 1024
    verify = tir.transform.VerifyGPUCode({"max_shared_memory_per_block": cap})

    with pytest.raises(tvm.error.TVMError):
        _ = verify(mod)


def test_pipeline_passcontext_verifies_shared_mem_overflow():
    mod = tvm.IRModule({"main": __tir_kernel_exceed_smem__})
    # Enable pipeline-level verification with a conservative cap
    with tvm.transform.PassContext(config={"tir.verify_gpu_code": True, "tir.cuda.max_shared_memory_per_block": 48 * 1024}):
        with pytest.raises(tvm.error.TVMError):
            # Building device code should trigger the verifier
            _ = tvm.tir.build(mod, target="cuda")


