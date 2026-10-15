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
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_rocm(), reason="need rocm")
def test_export_load_with_fallback(monkeypatch, tmp_path):
    """Force the codegen wrapper into the fallback branch, then export+load+run."""
    n = 1024

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", n // 32) as i_0:
                with T.launch_thread("threadIdx.x", 32) as i_1:
                    B[i_0 * 32 + i_1] = A[i_0 * 32 + i_1] + 1.0

    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    host_lib = tvm_tirx.tirx.compile(Module, target="rocm")
    monkeypatch.delenv("TVM_COMPILE_FORCE_FALLBACK")

    lib_path = str(tmp_path / "lib.so")
    host_lib.export_library(lib_path)
    reloaded = tvm_tirx.runtime.load_module(lib_path)

    a_np = np.random.uniform(size=(n,)).astype("float32")
    b_np = np.zeros((n,), dtype="float32")

    def run_and_check():
        dev = tvm_tirx.rocm(0)
        a = tvm_tirx.runtime.tensor(a_np, dev)
        b = tvm_tirx.runtime.tensor(b_np, dev)
        reloaded["main"](a, b)
        np.testing.assert_allclose(b.numpy(), a_np + 1.0, rtol=1e-5)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
