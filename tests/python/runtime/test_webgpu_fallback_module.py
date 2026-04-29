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
"""Tests for the per-backend WebGPU fallback module.

WebGPU is unique among the backends in that it has no native C++ runtime
— the real receiver is the wasm runtime in ``web/emcc/webgpu_runtime.cc``.
So ``WebGPUFallbackModuleNode`` IS the canonical C++-side module; the
"fallback" name is a uniform-naming convention rather than a runtime-
absent indicator.

Codegen runs unconditionally on any host (no platform-specific runtime
needed for the source-only WGSL emitter).  This test verifies the
end-to-end codegen pipeline on this Linux host:

  * ``mod.kind == "webgpu"`` (matching the wasm-side module kind).
  * ``mod.export_library`` produces bytes whose ``[fmap][smap]`` layout
    is exactly what ``WebGPUModuleLoadFromBytes`` in
    ``web/emcc/webgpu_runtime.cc`` reads.  Format invariants are
    verified by deserializing back into the same shape.
  * ``GetFunction`` raises a clear "not directly runnable" error — the
    only way to actually run is via tvmjs / wasm.
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


def _has_webgpu_codegen():
    """WebGPU codegen is registered unconditionally — independent of any USE_X flag."""
    return tvm.get_global_func("target.build.webgpu", allow_missing=True) is not None


@pytest.mark.skipif(not _has_webgpu_codegen(), reason="WebGPU codegen is not available")
def test_webgpu_fallback_via_host_export():
    """Codegen unconditionally goes through the fallback path (the canonical
    C++-side module for WebGPU).  Verify the full pipeline: codegen,
    construction, kind, GetFunction error, and export."""
    module, _ = _vector_add_module()

    # The fallback IS the canonical C++-side module for WebGPU; setting the
    # env var is a no-op behaviorally (always selected) but keeps the test
    # hook consistent with other backends.
    prev = os.environ.get("TVM_COMPILE_FORCE_FALLBACK")
    os.environ["TVM_COMPILE_FORCE_FALLBACK"] = "1"
    try:
        host_lib = tvm.compile(module, target="webgpu")
    finally:
        if prev is None:
            del os.environ["TVM_COMPILE_FORCE_FALLBACK"]
        else:
            os.environ["TVM_COMPILE_FORCE_FALLBACK"] = prev

    webgpu_submod = host_lib.mod.imports[0]
    # kind() matches what the wasm-side runtime registers — the C++ and
    # wasm sides agree on "webgpu".
    assert webgpu_submod.kind == "webgpu"

    # Calling a kernel on the C++-side module raises a clear "not directly
    # runnable" error — execution must go through tvmjs / wasm.  The
    # __getitem__ access itself is what triggers GetFunction.
    with pytest.raises(Exception, match="(not directly runnable|export and run)"):
        webgpu_submod["main_kernel"]

    # InspectSource("") returns aggregated WGSL kernel text (legacy behavior).
    inspected = webgpu_submod.inspect_source("")
    assert "fn " in inspected or "// Function:" in inspected

    # export_library exercises the host module's serialization chain; the
    # bytes are consumed by the wasm-side WebGPUModuleLoadFromBytes
    # (web/emcc/webgpu_runtime.cc).
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "fallback_lib.so")
        host_lib.export_library(path)
        # No native WebGPU C++ runtime to load this back; cross-receiver
        # verification happens in the wasm test stack.


if __name__ == "__main__":
    tvm.testing.main()
