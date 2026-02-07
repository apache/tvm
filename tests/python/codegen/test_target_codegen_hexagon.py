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
import pytest
import tvm
import tvm.testing
import tvm.contrib.hexagon as hexagon
from tvm.script import tir as T, ir as I


@pytest.fixture(autouse=True)
def register_linker():
    original_linker = hexagon.hexagon_link()
    # Register a phony linker, so that we can test codegen without a Hexagon toolchain.
    hexagon.register_linker(lambda: "/bin/true")
    yield None
    # Restore registration.
    hexagon.register_linker(original_linker)


@tvm.testing.requires_hexagon
def test_basic():
    target = tvm.target.hexagon("v66", hvx=128)

    @I.ir_module
    class Module:
        @T.prim_func
        def main(
            C: T.Buffer((128,), "uint8"),
            A: T.Buffer((128,), "uint8"),
            A_1: T.Buffer((128,), "uint8"),
        ):
            T.func_attr({"tir.noalias": True})
            for i in range(128):
                with T.sblock("C"):
                    v_i = T.axis.spatial(128, i)
                    T.reads(A[v_i], A_1[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] + A_1[v_i]

    hexm = tvm.compile(Module, target=tvm.target.Target(target, target))
    asm = hexm.inspect_source("s")
    vadds = re.findall(r"v[0-9]+.b = vadd\(v[0-9]+.b,v[0-9]+.b\)", asm)
    assert vadds  # Check that it's non-empty


@tvm.testing.requires_hexagon
def test_llvm_target_features():
    target = tvm.target.hexagon("v66", hvx=128)

    @I.ir_module
    class Module:
        @T.prim_func
        def add_one(C: T.Buffer((128,), "int32"), A: T.Buffer((128,), "uint8")):
            T.func_attr({"tir.noalias": True})
            for i in range(128):
                with T.sblock("C"):
                    v_i = T.axis.spatial(128, i)
                    T.reads(A[v_i])
                    T.writes(C[v_i])
                    C[v_i] = T.Cast("int32", A[v_i]) + 1

    m = tvm.compile(Module, target=tvm.target.Target(target, target))
    llvm_ir = m.inspect_source("ll")
    # Make sure we find +hvx-length128b in "attributes".
    fs = re.findall(r"attributes.*\+hvx-length128b", llvm_ir)
    assert fs  # Check that it's non-empty


@tvm.testing.requires_hexagon
def test_llvm_options():
    target = tvm.target.hexagon("v66", llvm_options="-hexagon-noopt")

    @I.ir_module
    class Module:
        @T.prim_func
        def main(compute: T.Buffer((10,), "int32")):
            T.func_attr({"tir.noalias": True})
            for _ in range(10):
                with T.sblock("compute"):
                    v__ = T.axis.spatial(10, _)
                    T.reads()
                    T.writes(compute[v__])
                    compute[v__] = 0

    # Check that BuildHexagon hasn't crashed because of target attribute
    # type mismatch.
    tvm.compile(Module, target=tvm.target.Target(target, target))
    assert re.search("-hexagon-noopt", str(target))


if __name__ == "__main__":
    tvm.testing.main()
