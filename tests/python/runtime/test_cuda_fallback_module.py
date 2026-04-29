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
"""End-to-end test for the per-backend CUDA fallback module.

The test forces the codegen-side wrapper into the fallback branch via the
``TVM_COMPILE_FORCE_FALLBACK`` env var, exercising:

    codegen wrapper (forced) -> fallback branch -> host module imports a
    CUDAFallbackModuleNode -> normal export_library -> normal load_module
    -> real loader JITs from source -> run + correctness assert.
"""

import os
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T


def _vector_add_module():
    n = 1024

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((n,), "float32"), B: T.Buffer((n,), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i_0 in T.thread_binding(n // 32, thread="blockIdx.x"):
                for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                    with T.sblock("B"):
                        v_i = T.axis.spatial(n, i_0 * 32 + i_1)
                        T.reads(A[v_i])
                        T.writes(B[v_i])
                        B[v_i] = A[v_i] + 1.0

    return Module, n


@tvm.testing.requires_cuda
def test_cuda_fallback_via_host_export(monkeypatch):
    """Force the codegen wrapper into the fallback branch, then export+load+run."""
    module, n = _vector_add_module()

    # 1. Force the wrapper into fallback branch for codegen.
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")

    # 2. Compile.  host_lib's cuda submodule is now a CUDAFallbackModuleNode.
    host_lib = tvm.compile(module, target="cuda")
    cuda_submod = host_lib.mod.imports[0]
    # kind() matches the real module — the fallback is indistinguishable at the
    # kind/api level.
    assert cuda_submod.kind == "cuda"
    # Calling a kernel on the fallback raises a clear "runtime not linked"
    # error — the only behavioral difference vs a real CUDAModuleNode.
    with pytest.raises(Exception, match="(runtime is not linked|cannot launch|not registered)"):
        cuda_submod["main_kernel"]()

    # 3. Unset env var so subsequent operations (export/load) take the normal path.
    monkeypatch.delenv("TVM_COMPILE_FORCE_FALLBACK")

    # 4. Normal export.  host_lib.export_library exercises the host module's
    #    serialization chain; the fallback's SaveToBytes produces bytes
    #    byte-identical to what a real CUDAModule would produce for the same
    #    payload.
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "fallback_lib.so")
        host_lib.export_library(path)

        # 5. Normal load.  The CUDA runtime IS registered on this CI, so
        #    ffi.Module.load_from_bytes.cuda dispatches to the real loader,
        #    which JITs from source if fmt=="cuda".
        reloaded = tvm.runtime.load_module(path)

        # 6. Run + assert correctness.
        dev = tvm.cuda(0)
        a_np = np.random.uniform(size=(n,)).astype("float32")
        b_np = np.zeros((n,), dtype="float32")
        a = tvm.runtime.tensor(a_np, dev)
        b = tvm.runtime.tensor(b_np, dev)
        reloaded["main"](a, b)
        np.testing.assert_allclose(b.numpy(), a_np + 1.0, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
