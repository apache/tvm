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

import re
from pathlib import Path

import pytest
import tvm_ffi
from tvm_ffi import register_global_func

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.hexagon.linker import has_hexagon_toolchain
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@pytest.fixture(autouse=True)
def register_linker():
    name = "tvm.contrib.hexagon.link_shared"
    original_linker = tvm_ffi.get_global_func(name)

    def link_shared(so_name, _objs, _extra_args=None):
        # Codegen retains the output bytes, but these tests only inspect the
        # LLVM-generated assembly/IR and never execute the linked module.
        Path(so_name).write_bytes(b"")
        return 0

    register_global_func(name, link_shared, override=True)
    try:
        yield
    finally:
        register_global_func(name, original_linker, override=True)


@pytest.mark.skipif(not has_hexagon_toolchain(), reason="need hexagon toolchain")
def test_basic():
    target = tvm_tirx.target.Target("qcom/hexagon-v66")

    @I.ir_module
    class Module:
        @T.function
        def main(
            C: T.Tensor((128,), "uint8"),
            A: T.Tensor((128,), "uint8"),
            A_1: T.Tensor((128,), "uint8"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i in range(128):
                C[i] = A[i] + A_1[i]

    hexm = tvm_tirx.tirx.compile(Module, target=tvm_tirx.target.Target(target, target))
    asm = hexm.mod.imports[0].inspect_source("s")
    vadds = re.findall(r"v[0-9]+.b = vadd\(v[0-9]+.b,v[0-9]+.b\)", asm)
    assert vadds, asm


@pytest.mark.skipif(not has_hexagon_toolchain(), reason="need hexagon toolchain")
def test_llvm_target_features():
    target = tvm_tirx.target.Target("qcom/hexagon-v66")

    @I.ir_module
    class Module:
        @T.function
        def add_one(C: T.Tensor((128,), "int32"), A: T.Tensor((128,), "uint8")):
            T.func_attr({"tirx.noalias": True})
            for i in range(128):
                C[i] = T.Cast("int32", A[i]) + 1

    m = tvm_tirx.tirx.compile(Module, target=tvm_tirx.target.Target(target, target))
    llvm_ir = m.mod.imports[0].inspect_source("ll")
    # Make sure we find +hvx-length128b in "attributes".
    fs = re.findall(r"attributes.*\+hvx-length128b", llvm_ir)
    assert fs  # Check that it's non-empty


@pytest.mark.skipif(not has_hexagon_toolchain(), reason="need hexagon toolchain")
def test_llvm_options():
    target = tvm_tirx.target.Target(
        {
            "kind": "hexagon",
            "mtriple": "hexagon",
            "mcpu": "hexagonv66",
            "mattr": ["+hvxv66", "+hvx-length128b"],
            "num-cores": 4,
            "vtcm-capacity": 262144,
            "llvm-options": ["-hexagon-noopt"],
        }
    )

    @I.ir_module
    class Module:
        @T.function
        def main(compute: T.Tensor((10,), "int32")):
            T.func_attr({"tirx.noalias": True})
            for _ in range(10):
                compute[_] = 0

    # Check that BuildHexagon hasn't crashed because of target attribute
    # type mismatch.
    tvm_tirx.tirx.compile(Module, target=tvm_tirx.target.Target(target, target))
    assert re.search("-hexagon-noopt", str(target))


if __name__ == "__main__":
    tvm_tirx.testing.main()
