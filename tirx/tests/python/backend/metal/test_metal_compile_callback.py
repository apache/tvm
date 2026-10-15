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
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_metal_compile_callback_source_passthrough():
    n = 1024

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", n // 32) as i_0:
                with T.launch_thread("threadIdx.x", 32) as i_1:
                    B[i_0 * 32 + i_1] = A[i_0 * 32 + i_1] + 1.0

    seen = {}

    def inspect_callback(src, target):
        # Pure inspection callback: capture the source, return it untouched and
        # declare it is still textual MSL so it is compiled at load time.
        seen["src"] = src
        return (src, "metal")

    tvm_tirx.register_global_func("tvm_callback_metal_compile", inspect_callback, override=True)
    try:
        f = tvm_tirx.tirx.compile(Module, target="metal")
        dev = tvm_tirx.metal()
        a = np.random.rand(n).astype("float32")
        a_nd = tvm_tirx.runtime.tensor(a, dev)
        b_nd = tvm_tirx.runtime.empty((n,), "float32", dev)
        f(a_nd, b_nd)
        dev.sync()
    finally:
        tvm_ffi.registry.remove_global_func("tvm_callback_metal_compile")

    assert "src" in seen and len(seen["src"]) > 0
    tvm_tirx.testing.assert_allclose(b_nd.numpy(), a + 1.0, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_metal_compile_callback_mixed_formats_rejected():
    n = 1024

    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((n,), "float32"),
            B: T.Tensor((n,), "float32"),
            C: T.Tensor((n,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # Two independent thread-bound regions -> two device kernels, so the
            # compile callback is invoked twice within one module.
            with T.launch_thread("blockIdx.x", n // 32) as i_0:
                with T.launch_thread("threadIdx.x", 32) as i_1:
                    B[i_0 * 32 + i_1] = A[i_0 * 32 + i_1] + 1.0
            with T.launch_thread("blockIdx.x", n // 32) as j_0:
                with T.launch_thread("threadIdx.x", 32) as j_1:
                    C[j_0 * 32 + j_1] = A[j_0 * 32 + j_1] + 2.0

    calls = {"n": 0}

    def mixed_callback(src, target):
        calls["n"] += 1
        if calls["n"] == 1:
            # Treated as a compiled metallib payload.
            return src
        # Second kernel declares textual MSL, contradicting the metallib above.
        return (src, "metal")

    tvm_tirx.register_global_func("tvm_callback_metal_compile", mixed_callback, override=True)
    try:
        with pytest.raises(Exception, match="inconsistent formats"):
            tvm_tirx.tirx.compile(Module, target="metal")
    finally:
        tvm_ffi.registry.remove_global_func("tvm_callback_metal_compile")


if __name__ == "__main__":
    tvm_tirx.testing.main()
