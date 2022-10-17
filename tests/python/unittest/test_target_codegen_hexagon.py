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
import os
import pytest
import re
import sys
import tvm
import tvm.relay
import tvm.testing
import tvm.contrib.hexagon as hexagon


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

    def check_add(offload):
        A = tvm.te.placeholder((128,), dtype="uint8", name="A")
        B = tvm.te.placeholder((128,), dtype="uint8", name="A")
        C = tvm.te.compute((128,), lambda i: A[i] + B[i], name="C")
        s = tvm.te.create_schedule(C.op)

        if offload:
            xo, xi = s[C].split(s[C].op.axis[0], nparts=1)
            s[C].bind(xo, tvm.te.thread_axis("pipeline"))
            m = tvm.build(s, [C, A, B], target=target, name="offload_add")
            hexm = m.imported_modules[0]
        else:
            hexm = tvm.build(
                s, [C, A, B], target=tvm.target.Target(target, target), name="native_add"
            )

        asm = hexm.get_source("s")
        vadds = re.findall(r"v[0-9]+.b = vadd\(v[0-9]+.b,v[0-9]+.b\)", asm)
        assert vadds  # Check that it's non-empty

    check_add(True)
    check_add(False)


@tvm.testing.requires_hexagon
def test_llvm_target_features():
    target = tvm.target.hexagon("v66", hvx=128)
    # Define some trivial compute
    A = tvm.te.placeholder((128,), dtype="uint8", name="A")
    C = tvm.te.compute((128,), lambda i: A[i] + 1, name="C")
    s = tvm.te.create_schedule(C.op)
    m = tvm.build(s, [C, A], target=tvm.target.Target(target, target), name="add_one")
    llvm_ir = m.get_source("ll")
    # Make sure we find +hvx-length128b in "attributes".
    fs = re.findall(r"attributes.*\+hvx-length128b", llvm_ir)
    assert fs  # Check that it's non-empty


@tvm.testing.requires_hexagon
def test_alloc_vtcm():
    target = tvm.target.hexagon("v66")

    buf_len = 2048
    A = tvm.te.placeholder((buf_len,), name="A", dtype="int8")
    B = tvm.te.placeholder((buf_len,), name="B", dtype="int8")

    A_buf = tvm.te.compute((buf_len,), lambda *i: A(*i), "A_buf")
    B_buf = tvm.te.compute((buf_len,), lambda *i: B(*i), "B_buf")
    C = tvm.te.compute((buf_len,), lambda *i: A_buf(*i) + B_buf(*i), name="C")
    s = tvm.te.create_schedule(C.op)

    # Use VTCM for each buffer.
    s[A_buf].set_scope("local.vtcm")
    s[B_buf].set_scope("local.vtcm")

    config = {"tir.add_lower_pass": hexagon.ir_lower_vtcm_pass()}
    with tvm.transform.PassContext(config=config):
        irmod = tvm.lower(s, [A, B, C], name="alloc_vtcm")

    calls = re.findall("HexagonBackend[A-Za-z]*VTCM", str(irmod["alloc_vtcm"]))
    assert "HexagonBackendAllocateVTCM" in calls
    assert "HexagonBackendFreeVTCM" in calls


@tvm.testing.requires_hexagon
def test_llvm_options():
    target = tvm.target.hexagon("v66", llvm_options="-hexagon-noopt")
    Zero = tvm.te.compute((10,), lambda _: tvm.tir.const(0, "int32"))
    s = tvm.te.create_schedule(Zero.op)
    tvm.build(s, [Zero], target=target, name="zero")
    # Check that BuildHexagon hasn't crashed because of target attribute
    # type mismatch.
    assert re.search("-hexagon-noopt", str(target))


if __name__ == "__main__":
    tvm.testing.main()
