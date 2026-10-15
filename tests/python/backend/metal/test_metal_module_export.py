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
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.testing import env


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_export_load_with_fallback(monkeypatch, tmp_path):
    """Force the codegen wrapper into the fallback branch, then export."""
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
    host_lib = tvm.tirx.compile(Module, target="metal")
    monkeypatch.delenv("TVM_COMPILE_FORCE_FALLBACK")

    lib_path = str(tmp_path / "lib.so")
    host_lib.export_library(lib_path)


if __name__ == "__main__":
    tvm.testing.main()
