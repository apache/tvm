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
"""Tests for the per-backend Vulkan fallback module.

Vulkan codegen produces SPIR-V binaries via the SPIR-V tooling under
``src/target/vulkan/`` (absorbed from the legacy ``src/target/spirv/``).
The Vulkan *runtime* requires Vulkan-capable hardware/drivers and is
gated on ``USE_VULKAN=ON``.

The fallback path: when the runtime FFI keys are absent (USE_VULKAN=OFF),
or when ``TVM_COMPILE_FORCE_FALLBACK`` is set, codegen routes through
``VulkanModuleCreateWithFallback`` which constructs a
``VulkanFallbackModuleNode``.  Saved-bytes are byte-identical to a real
``VulkanModuleNode`` for the same payload — a Vulkan-equipped receiver
can load the same bytes via ``ffi.Module.load_from_bytes.vulkan``.

This test exercises the full codegen→fallback→export pipeline whenever
the Vulkan codegen is available (i.e. SPIR-V tooling is linked, which is
gated on ``USE_VULKAN=ON`` per ``cmake/modules/Vulkan.cmake``).  When
``USE_VULKAN=OFF``, the test verifies the fallback-on-miss precondition
(create-/load-bytes keys absent) instead.
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


def _has_vulkan_codegen():
    """Vulkan codegen requires SPIR-V tooling — gated on USE_VULKAN=ON
    per cmake/modules/Vulkan.cmake."""
    return tvm.get_global_func("target.build.vulkan", allow_missing=True) is not None


def _vulkan_runtime_loader_registered():
    """The real Vulkan runtime registers ffi.Module.load_from_bytes.vulkan
    when USE_VULKAN=ON; both codegen and runtime live in the same
    Vulkan.cmake gate, so this is equivalent to ``_has_vulkan_codegen``
    on this build."""
    return tvm.get_global_func("ffi.Module.load_from_bytes.vulkan", allow_missing=True) is not None


@pytest.mark.skipif(not _has_vulkan_codegen(), reason="Vulkan codegen is not available")
def test_vulkan_fallback_via_host_export():
    """Codegen with TVM_COMPILE_FORCE_FALLBACK=1 always routes through the
    fallback path even when the real runtime is registered."""
    module, _ = _vector_add_module()

    # Force fallback so the test is deterministic on USE_VULKAN=ON hosts.
    prev = os.environ.get("TVM_COMPILE_FORCE_FALLBACK")
    os.environ["TVM_COMPILE_FORCE_FALLBACK"] = "1"
    try:
        host_lib = tvm.compile(module, target="vulkan")
    finally:
        if prev is None:
            del os.environ["TVM_COMPILE_FORCE_FALLBACK"]
        else:
            os.environ["TVM_COMPILE_FORCE_FALLBACK"] = prev

    vulkan_submod = host_lib.mod.imports[0]
    # kind() matches the real module — the fallback is indistinguishable
    # at the kind/api level.
    assert vulkan_submod.kind == "vulkan"

    # Calling a kernel on the fallback raises a clear "runtime not linked"
    # error — the only behavioral difference vs a real VulkanModuleNode.
    with pytest.raises(Exception, match="(runtime is not linked|cannot launch|not registered)"):
        vulkan_submod["main_kernel"]()

    # export_library exercises the host module's serialization chain; the
    # fallback's SaveToBytes produces bytes byte-identical to what a real
    # VulkanModuleNode would produce for the same payload.
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "fallback_lib.so")
        host_lib.export_library(path)
        # Receiving env (USE_VULKAN=ON Vulkan-equipped box) would
        # `tvm.runtime.load_module(path)` to JIT real shader modules;
        # not exercised here because that requires Vulkan drivers/devices.


@pytest.mark.skipif(
    _vulkan_runtime_loader_registered(),
    reason="USE_VULKAN=ON in this build — fallback-on-miss precondition does not hold",
)
def test_vulkan_fallback_registry_preconditions():
    """On USE_VULKAN=OFF, the create- and load-keys are absent, which is
    the precondition for the fallback-on-miss path."""
    assert tvm.get_global_func("ffi.Module.create.vulkan", allow_missing=True) is None
    assert tvm.get_global_func("ffi.Module.load_from_bytes.vulkan", allow_missing=True) is None


if __name__ == "__main__":
    tvm.testing.main()
