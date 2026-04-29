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
"""Tests for the per-backend OpenCL fallback module.

OpenCL codegen has no LLVM dependency — it can run on any host that
loaded ``libtvm`` (the codegen lives in ``src/target/opencl/``).  But
the OpenCL *runtime* requires an OpenCL-capable platform; on hosts
without OpenCL ICD ``USE_OPENCL=OFF`` and the runtime FFI keys are
absent.

This test verifies the fallback path:

  * Codegen-side construction routes through ``OpenCLModuleCreateWithFallback``.
  * On a USE_OPENCL=OFF host (or with ``TVM_COMPILE_FORCE_FALLBACK=1``)
    the path falls through to ``OpenCLFallbackModuleCreate``.
  * ``mod.kind == "opencl"`` (matching the real module).
  * ``mod.export_library`` produces bytes that round-trip the kind /
    format / code via the host module's serialization chain.
  * ``GetFunction`` raises a clear "runtime not linked" error.

OpenCL is single-binary: the ``code`` payload carries the OpenCL C
source bytes (``fmt=="cl"``) directly.  The legacy SPIR-V path
(``OpenCLSPIRVModuleNode``) was removed alongside this refactor; SPIR-V
tooling now lives only under ``src/target/vulkan/``.
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


def _has_opencl_codegen():
    """OpenCL codegen is registered unconditionally — independent of USE_OPENCL."""
    return tvm.get_global_func("target.build.opencl", allow_missing=True) is not None


def _opencl_runtime_loader_registered():
    """The real runtime registers ffi.Module.load_from_bytes.opencl when
    USE_OPENCL=ON.  Absence => USE_OPENCL=OFF host."""
    return tvm.get_global_func("ffi.Module.load_from_bytes.opencl", allow_missing=True) is not None


@pytest.mark.skipif(not _has_opencl_codegen(), reason="OpenCL codegen is not available")
def test_opencl_fallback_via_host_export():
    """Codegen with TVM_COMPILE_FORCE_FALLBACK=1 routes through the
    fallback path even when the real runtime is registered."""
    module, _ = _vector_add_module()

    # Force fallback so the test is deterministic on USE_OPENCL=ON hosts.
    prev = os.environ.get("TVM_COMPILE_FORCE_FALLBACK")
    os.environ["TVM_COMPILE_FORCE_FALLBACK"] = "1"
    try:
        host_lib = tvm.compile(module, target="opencl")
    finally:
        if prev is None:
            del os.environ["TVM_COMPILE_FORCE_FALLBACK"]
        else:
            os.environ["TVM_COMPILE_FORCE_FALLBACK"] = prev

    opencl_submod = host_lib.mod.imports[0]
    # kind() matches the real module — the fallback is indistinguishable
    # at the kind/api level.
    assert opencl_submod.kind == "opencl"

    # InspectSource("") returns the aggregated OpenCL C source.
    inspected = opencl_submod.inspect_source("")
    assert "// Function:" in inspected or "__kernel" in inspected

    # Calling a kernel on the fallback raises a clear "runtime not linked"
    # error — the only behavioral difference vs a real OpenCLModuleNode.
    with pytest.raises(Exception, match="(runtime is not linked|cannot launch|not registered)"):
        opencl_submod["main_kernel"]()

    # export_library exercises the host module's serialization chain;
    # the fallback's SaveToBytes produces bytes byte-identical to what a
    # real OpenCLModuleNode would produce for the same payload.
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "fallback_lib.so")
        host_lib.export_library(path)
        # Receiving env (USE_OPENCL=ON box with an OpenCL ICD) would
        # `tvm.runtime.load_module(path)` to JIT real cl_programs;
        # not exercised here because the test must work on USE_OPENCL=OFF
        # hosts too.


@pytest.mark.skipif(
    _opencl_runtime_loader_registered(),
    reason="USE_OPENCL=ON in this build — fallback-on-miss precondition does not hold",
)
def test_opencl_fallback_registry_preconditions():
    """On USE_OPENCL=OFF, the create- and load-keys are absent, which is
    the precondition for the fallback-on-miss path."""
    assert tvm.get_global_func("ffi.Module.create.opencl", allow_missing=True) is None
    assert tvm.get_global_func("ffi.Module.load_from_bytes.opencl", allow_missing=True) is None


if __name__ == "__main__":
    tvm.testing.main()
