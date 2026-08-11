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

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T


def test_codegen_buffer_access_modes():
    """Read-only typed buffer parameters should remain read-only in WGSL."""

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer((8,), "float32"), B: T.Buffer((8,), "float32")):
            for tx in T.thread_binding(8, thread="threadIdx.x"):
                B[tx] = A[tx]

    executable = tvm.compile(Module, target="webgpu")
    source = executable.mod.imports[0].inspect_source("wgsl")

    assert "var<storage, read> A_ptr" in source
    assert "var<storage, read_write> B_ptr" in source


if __name__ == "__main__":
    tvm.testing.main()
