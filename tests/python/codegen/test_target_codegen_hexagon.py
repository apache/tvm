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

import os
import re
import sys
import numpy as np
import pytest
import tvm
import tvm.testing
import tvm.contrib.hexagon as hexagon
from tvm import te


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

    def check_add():
        A = tvm.te.placeholder((128,), dtype="uint8", name="A")
        B = tvm.te.placeholder((128,), dtype="uint8", name="A")
        C = tvm.te.compute((128,), lambda i: A[i] + B[i], name="C")
        mod = tvm.IRModule.from_expr(te.create_prim_func([C, A, B]))
        hexm = tvm.compile(mod, target=tvm.target.Target(target, target))
        asm = hexm.get_source("s")
        vadds = re.findall(r"v[0-9]+.b = vadd\(v[0-9]+.b,v[0-9]+.b\)", asm)
        assert vadds  # Check that it's non-empty

    check_add()


@tvm.testing.requires_hexagon
def test_llvm_target_features():
    target = tvm.target.hexagon("v66", hvx=128)
    # Define some trivial compute
    A = tvm.te.placeholder((128,), dtype="uint8", name="A")
    C = tvm.te.compute((128,), lambda i: A[i] + 1, name="C")
    mod = tvm.IRModule.from_expr(te.create_prim_func([C, A]).with_attr("global_symbol", "add_one"))
    m = tvm.compile(mod, target=tvm.target.Target(target, target))
    llvm_ir = m.get_source("ll")
    # Make sure we find +hvx-length128b in "attributes".
    fs = re.findall(r"attributes.*\+hvx-length128b", llvm_ir)
    assert fs  # Check that it's non-empty


@tvm.testing.requires_hexagon
def test_llvm_options():
    target = tvm.target.hexagon("v66", llvm_options="-hexagon-noopt")
    Zero = tvm.te.compute((10,), lambda _: tvm.tir.const(0, "int32"))
    mod = tvm.IRModule.from_expr(te.create_prim_func([Zero]))
    # Check that BuildHexagon hasn't crashed because of target attribute
    # type mismatch.
    tvm.compile(mod, target=tvm.target.Target(target, target))
    assert re.search("-hexagon-noopt", str(target))


if __name__ == "__main__":
    tvm.testing.main()
