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
import tvm
import tvm.contrib.hexagon as hexagon


def check_prereq_and_setup():
    if tvm.target.codegen.llvm_version_major() <= 7:
        print("Skipping test: need LLVM 7 or later for codegen")
        return False
    if os.name != "posix":
        print("Skipping test on non-POSIX platforms")
        return False
    if not tvm.runtime.enabled("hexagon"):
        print("Hexagon runtime not enabled")
        return False
    # Register a phony linker, so that we can test codegen without a Hexagon toolchain.
    hexagon.register_linker(lambda: "/bin/true")
    return True


def test_basic():
    if not check_prereq_and_setup():
        return
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
            hexm = tvm.build(s, [C, A, B], target=target, target_host=target, name="native_add")

        asm = hexm.get_source("s")
        vadds = re.findall(r"v[0-9]+.b = vadd\(v[0-9]+.b,v[0-9]+.b\)", asm)
        assert vadds  # Check that it's non-empty

    check_add(True)
    check_add(False)


def test_alloc_vtcm():
    if not check_prereq_and_setup():
        return
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


if __name__ == "__main__":
    test_basic()
    test_alloc_vtcm()
