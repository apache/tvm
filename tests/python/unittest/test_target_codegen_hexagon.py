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
import re
import tvm
import tvm.relay
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
            hexm = tvm.build(
                s, [C, A, B], target=tvm.target.Target(target, target), name="native_add"
            )

        asm = hexm.get_source("s")
        vadds = re.findall(r"v[0-9]+.b = vadd\(v[0-9]+.b,v[0-9]+.b\)", asm)
        assert vadds  # Check that it's non-empty

    check_add(True)
    check_add(False)


def test_llvm_target_features():
    if not check_prereq_and_setup():
        return
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


def test_linked_params_codegen():
    if not check_prereq_and_setup():
        return

    # A simple model (a single conv2d) to trigger parameter separation:
    mod_lines = [
        '#[version = "0.0.5"]',
        "def @main(%input: Tensor[(1, 16, 16, 3), uint8], %weights: Tensor[(3, 3, 3, 3), uint8])"
        " -> Tensor[(1, 14, 14, 3), uint8] {",
        '  nn.conv2d(%input, %weights, data_layout="NHWC", kernel_layout="HWIO", '
        'kernel_size=[3, 3], out_dtype="uint8")',
        "}",
    ]
    mod = tvm.parser.fromtext("\n".join(mod_lines))
    # Make the params be 81 x 'T':
    params = {"weights": np.full([3, 3, 3, 3], fill_value=ord("T"), dtype=np.uint8)}

    target = tvm.target.hexagon("v68", link_params=True)

    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(mod, target=target, target_host=target, params=params)
        llvm_ir = lib.get_lib().get_source("ll")

    # The definition of the parameter:
    p0_def_re = r"@__tvm_param__p0 = internal constant \[81 x i8\] c\"T{81}\", align 128"
    assert re.search(p0_def_re, llvm_ir)

    # The body of the _lookup_linked_param function:
    linked_param_re = r"(define.*@_lookup_linked_param\(.*\).* {[^}]*})"
    linked_param_body = re.search(linked_param_re, llvm_ir, flags=re.MULTILINE)
    assert linked_param_body and linked_param_body.groups()

    # Reference to the parameter:
    p0_use_re = r"\[81 x i8\]\* @__tvm_param__p0"
    assert re.search(p0_use_re, linked_param_body.groups()[0])

    """
    A snippet of actual LLVM IR containing the definition of the linked
    parameter, and the the body of the _lookup_linked_param function.


    @__tvm_param__p0 = internal constant [81 x i8] c"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT", align 128

    define dllexport i32 @_lookup_linked_param(i8* nocapture readonly %0, i32* nocapture readnone %1, i32 %2, i8* nocapture %3, i32* nocapture %4, i8* nocapture readnone %5) local_unnamed_addr #2 {
    entry:
      %6 = bitcast i8* %0 to i64*
      %7 = load i64, i64* %6, align 8
      %cond = icmp eq i64 %7, 1
      br i1 %cond, label %case___tvm_param__p0, label %common.ret

    common.ret:                                       ; preds = %entry, %case___tvm_param__p0
      %storemerge = phi i32 [ 3, %case___tvm_param__p0 ], [ 4, %entry ]
      store i32 %storemerge, i32* %4, align 4
      ret i32 0

    case___tvm_param__p0:                             ; preds = %entry
      %8 = bitcast i8* %3 to i8**
      store i8* getelementptr inbounds ([81 x i8], [81 x i8]* @__tvm_param__p0, i32 0, i32 0), i8** %8, align 4
      br label %common.ret
    }
    """


if __name__ == "__main__":
    test_basic()
    test_llvm_target_features()
    test_alloc_vtcm()
    test_linked_params_codegen()
