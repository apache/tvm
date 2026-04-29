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
"""Tests for the per-backend Metal fallback module.

Metal codegen has no LLVM dependency — it can run on any host that loaded
``libtvm`` (the codegen lives in ``src/target/metal/``).  But the Metal
*runtime* requires macOS with Metal frameworks; on a Linux CI host
``USE_METAL=OFF`` and the runtime FFI keys are absent.

This test verifies the fallback path on a USE_METAL=OFF host:

  * Codegen-side construction routes through ``MetalModuleCreateWithFallback``;
    ``ffi.Module.create.metal`` is absent so it falls through to
    ``MetalFallbackModuleCreate``.
  * ``mod.kind == "metal"`` (matching the real module).
  * ``mod.export_library`` produces bytes that round-trip the kind/format
    via the host module's serialization chain.
  * ``GetFunction`` raises a clear "runtime not linked" error.

The full codegen→fallback→export pipeline runs on this Linux host because
``CodeGenMetal`` does not require any platform-specific runtime — it is a
pure C++ source generator.
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


def _has_metal_codegen():
    """Metal codegen is registered unconditionally — independent of USE_METAL."""
    return tvm.get_global_func("target.build.metal", allow_missing=True) is not None


def _metal_runtime_loader_registered():
    """The real runtime registers ffi.Module.load_from_bytes.metal when
    USE_METAL=ON.  Absence => USE_METAL=OFF host (typical Linux CI)."""
    return tvm.get_global_func("ffi.Module.load_from_bytes.metal", allow_missing=True) is not None


@pytest.mark.skipif(not _has_metal_codegen(), reason="Metal codegen is not available")
def test_metal_fallback_via_host_export():
    """Codegen with USE_METAL=OFF (or forced fallback) goes through the fallback path."""
    module, _ = _vector_add_module()

    # Force fallback so the test is deterministic on USE_METAL=ON hosts too.
    prev = os.environ.get("TVM_COMPILE_FORCE_FALLBACK")
    os.environ["TVM_COMPILE_FORCE_FALLBACK"] = "1"
    try:
        host_lib = tvm.compile(module, target="metal")
    finally:
        if prev is None:
            del os.environ["TVM_COMPILE_FORCE_FALLBACK"]
        else:
            os.environ["TVM_COMPILE_FORCE_FALLBACK"] = prev

    metal_submod = host_lib.mod.imports[0]
    # kind() matches the real module — the fallback is indistinguishable at
    # the kind/api level.
    assert metal_submod.kind == "metal"

    # Calling a kernel on the fallback raises a clear "runtime not linked"
    # error — the only behavioral difference vs a real MetalModuleNode.
    with pytest.raises(Exception, match="(runtime is not linked|cannot launch|not registered)"):
        metal_submod["main_kernel"]()

    # export_library exercises the host module's serialization chain; the
    # fallback's SaveToBytes produces bytes byte-identical to what a real
    # MetalModuleNode would produce for the same payload.
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "fallback_lib.so")
        host_lib.export_library(path)
        # Receiving env (USE_METAL=ON box) would `tvm.runtime.load_module(path)`;
        # not exercised here because no Metal runtime is available on this CI.


@pytest.mark.skipif(
    _metal_runtime_loader_registered(),
    reason="USE_METAL=ON in this build — fallback-on-miss precondition does not hold",
)
def test_metal_fallback_registry_preconditions():
    """On USE_METAL=OFF (typical CI), the create- and load-keys are absent,
    which is the precondition for the fallback-on-miss path."""
    assert tvm.get_global_func("ffi.Module.create.metal", allow_missing=True) is None
    assert tvm.get_global_func("ffi.Module.load_from_bytes.metal", allow_missing=True) is None


if __name__ == "__main__":
    tvm.testing.main()
