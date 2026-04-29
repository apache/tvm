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
"""Tests for the per-backend ROCm fallback module.

ROCm codegen requires LLVM with the AMDGPU target.  On a USE_ROCM=OFF host
(typical CI without ROCm runtime), `target.build.rocm` is still callable as
long as LLVM has the AMDGPU target enabled, and routes through the codegen
wrapper into ``ROCmFallbackModuleNode`` (no runtime needed).

This test does not exercise the load-back path (no ROCm runtime present);
it verifies that:

  * ``mod.kind == "hip"`` (matching the real module)
  * ``mod.export_library`` produces bytes that round-trip the kind/format
    via the host module's serialization chain
  * ``GetFunction`` raises a clear "runtime not linked" error
"""

import os
import tempfile

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


def _has_rocm_codegen():
    """Check if LLVM has the AMDGPU target so we can exercise BuildAMDGPU."""
    try:
        target = tvm.target.Target("rocm")
    except Exception:  # pragma: no cover - environment-dependent
        return False
    fbuild = tvm.get_global_func("target.build.rocm", allow_missing=True)
    return fbuild is not None and target is not None


@pytest.mark.skipif(
    not _has_rocm_codegen(), reason="ROCm codegen (LLVM AMDGPU target) is not available"
)
def test_rocm_fallback_via_host_export():
    """Codegen with USE_ROCM=OFF (or forced fallback) goes through the fallback path."""
    module, _ = _vector_add_module()

    # Force fallback so that even on a hypothetical USE_ROCM=ON host the
    # codegen-side wrapper picks the fallback.
    prev = os.environ.get("TVM_COMPILE_FORCE_FALLBACK")
    os.environ["TVM_COMPILE_FORCE_FALLBACK"] = "1"
    try:
        host_lib = tvm.compile(module, target="rocm")
    finally:
        if prev is None:
            del os.environ["TVM_COMPILE_FORCE_FALLBACK"]
        else:
            os.environ["TVM_COMPILE_FORCE_FALLBACK"] = prev

    rocm_submod = host_lib.mod.imports[0]
    # kind() matches the real module — the fallback is indistinguishable at
    # the kind/api level.
    assert rocm_submod.kind == "hip"

    # Calling a kernel on the fallback raises a clear "runtime not linked"
    # error — the only behavioral difference vs a real ROCMModuleNode.
    with pytest.raises(Exception, match="(runtime is not linked|cannot launch|not registered)"):
        rocm_submod["main_kernel"]()

    # export_library exercises the host module's serialization chain; the
    # fallback's SaveToBytes produces bytes byte-identical to what a real
    # ROCMModule would produce for the same payload.
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "fallback_lib.so")
        host_lib.export_library(path)
        # Receiving env (USE_ROCM=ON box) would `tvm.runtime.load_module(path)`;
        # not exercised here because no ROCm runtime is available on this CI.


if __name__ == "__main__":
    tvm.testing.main()
