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
import collections
import ctypes
import json
import sys

import tvm
import tvm.testing
from tvm import te
from tvm import topi
from tvm.contrib import utils
import numpy as np
import ctypes
import math
import re
import pytest


@tvm.testing.requires_llvm
def test_llvm_intrin():
    ib = tvm.tir.ir_builder.create()
    n = tvm.runtime.convert(4)
    A = ib.pointer("float32", name="A")
    args = [tvm.tir.call_intrin("handle", "tir.address_of", A[0]), 0, 3, 1]
    ib.emit(tvm.tir.Evaluate(tvm.tir.Call("int32", "tir.prefetch", args)))
    body = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A], body).with_attr("global_symbol", "prefetch"))
    fcode = tvm.build(mod, None, "llvm")


@tvm.testing.requires_llvm
def test_llvm_void_intrin():
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("uint8", name="A")
    # Create an intrinsic that returns void.
    x = tvm.tir.call_llvm_intrin("", "llvm.va_start", tvm.tir.const(1, "uint32"), A)
    ib.emit(x)
    body = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A], body).with_attr("global_symbol", "main"))
    fcode = tvm.build(mod, None, "llvm")


@tvm.testing.requires_llvm
def test_llvm_overloaded_intrin():
    # Name lookup for overloaded intrinsics in LLVM 4- requires a name
    # that includes the overloaded types.
    if tvm.target.codegen.llvm_version_major() < 5:
        return

    def use_llvm_intrinsic(A, C):
        ib = tvm.tir.ir_builder.create()
        L = A.vload((0, 0))
        I = tvm.tir.call_llvm_pure_intrin(
            "int32", "llvm.ctlz", tvm.tir.const(2, "uint32"), L, tvm.tir.const(0, "int1")
        )
        S = C.vstore((0, 0), I)
        ib.emit(S)
        return ib.get()

    A = tvm.te.placeholder((1, 1), dtype="int32", name="A")
    C = tvm.te.extern(
        (1, 1), [A], lambda ins, outs: use_llvm_intrinsic(ins[0], outs[0]), name="C", dtype="int32"
    )
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], target="llvm")


@tvm.testing.requires_llvm
def test_llvm_lookup_intrin():
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("uint8x8", name="A")
    z = tvm.tir.const(0, "int32")
    x = tvm.tir.call_llvm_pure_intrin(
        "uint8x8", "llvm.ctpop.v8i8", tvm.tir.const(1, "uint32"), A[z]
    )
    ib.emit(x)
    body = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A], body).with_attr("global_symbol", "main"))
    fcode = tvm.build(mod, None, "llvm")


@tvm.testing.requires_llvm
def test_llvm_large_uintimm():
    value = (1 << 63) + 123
    other = tvm.tir.const(3, "uint64")
    A = te.compute((), lambda: tvm.tir.const(value, "uint64") + other, name="A")
    s = te.create_schedule(A.op)

    def check_llvm():
        f = tvm.build(s, [A], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.empty((), dtype=A.dtype, ctx=ctx)
        f(a)
        assert a.asnumpy() == value + 3

    check_llvm()


@tvm.testing.requires_llvm
def test_llvm_persist_parallel():
    n = 128
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1, name="B")
    C = te.compute(A.shape, lambda *i: te.sqrt(B(*i)) * 2 + 2, name="C")
    s = te.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=8)
    xo1, xo2 = s[C].split(xo, nparts=1)
    s[B].compute_at(s[C], xo1)
    s[B].parallel(s[B].op.axis[0])
    s[B].pragma(s[B].op.axis[0], "parallel_barrier_when_finish")
    s[C].parallel(xi)
    s[C].pragma(xo1, "parallel_launch_point")
    s[C].pragma(xi, "parallel_stride_pattern")

    def check_llvm():
        # BUILD and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        f(a, c)
        tvm.testing.assert_allclose(c.asnumpy(), np.sqrt(a.asnumpy() + 1) * 2 + 2, rtol=1e-5)

    check_llvm()


@tvm.testing.requires_llvm
def test_llvm_flip_pipeline():
    def check_llvm(nn, base):
        n = tvm.runtime.convert(nn)
        A = te.placeholder((n + base), name="A")
        C = te.compute((n,), lambda i: A(nn + base - i - 1), name="C")
        s = te.create_schedule(C.op)
        xo, xi = s[C].split(C.op.axis[0], factor=4)
        s[C].parallel(xo)
        s[C].vectorize(xi)
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=(n + base)).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        f(a, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy()[::-1][:n])

    check_llvm(4, 0)
    check_llvm(128, 8)
    check_llvm(3, 0)
    check_llvm(128, 1)


@tvm.testing.requires_llvm
def test_llvm_vadd_pipeline():
    def check_llvm(n, lanes):
        A = te.placeholder((n,), name="A", dtype="float32x%d" % lanes)
        B = te.compute((n,), lambda i: A[i], name="B")
        C = te.compute((n,), lambda i: B[i] + tvm.tir.const(1, A.dtype), name="C")
        s = te.create_schedule(C.op)
        xo, xi = s[C].split(C.op.axis[0], nparts=2)
        _, xi = s[C].split(xi, factor=2)
        s[C].parallel(xo)
        s[C].vectorize(xi)
        s[B].compute_at(s[C], xo)
        xo, xi = s[B].split(B.op.axis[0], factor=2)
        s[B].vectorize(xi)
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.empty((n,), A.dtype).copyfrom(np.random.uniform(size=(n, lanes)))
        c = tvm.nd.empty((n,), C.dtype, ctx)
        f(a, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1)

    check_llvm(64, 2)
    check_llvm(512, 2)


@tvm.testing.requires_llvm
def test_llvm_madd_pipeline():
    def check_llvm(nn, base, stride):
        n = tvm.runtime.convert(nn)
        A = te.placeholder((n + base, stride), name="A")
        C = te.compute((n, stride), lambda i, j: A(base + i, j) + 1, name="C")
        s = te.create_schedule(C.op)
        xo, xi = s[C].split(C.op.axis[0], factor=4)
        s[C].parallel(xo)
        s[C].vectorize(xi)
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=(n + base, stride)).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros((n, stride), dtype=C.dtype), ctx)
        f(a, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy()[base:] + 1)

    check_llvm(64, 0, 2)
    check_llvm(4, 0, 1)

    with tvm.transform.PassContext(config={"tir.noalias": False}):
        check_llvm(4, 0, 3)


@tvm.testing.requires_llvm
def test_llvm_temp_space():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda i: A(i) + 1, name="B")
    C = te.compute(A.shape, lambda i: B(i) + 1, name="C")
    s = te.create_schedule(C.op)

    def check_llvm():
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        f(a, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1 + 1)

    check_llvm()


@tvm.testing.requires_llvm
def test_multiple_func():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = te.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    s[C].parallel(xo)
    s[C].vectorize(xi)

    def check_llvm():
        # build two functions
        f2 = tvm.lower(s, [A, B, C], name="fadd1")
        f1 = tvm.lower(s, [A, B, C], name="fadd2")
        m = tvm.build([f1, f2], "llvm")
        fadd2 = m["fadd2"]
        fadd1 = m["fadd1"]

        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        fadd1(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
        fadd2(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

    check_llvm()


@tvm.testing.requires_llvm
def test_llvm_condition():
    def check_llvm(n, offset):
        A = te.placeholder((n,), name="A")
        C = te.compute((n,), lambda i: tvm.tir.if_then_else(i >= offset, A[i], 0.0), name="C")
        s = te.create_schedule(C.op)
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), ctx)
        c = tvm.nd.empty((n,), A.dtype, ctx)
        f(a, c)
        c_np = a.asnumpy()
        c_np[:offset] = 0
        tvm.testing.assert_allclose(c.asnumpy(), c_np)

    check_llvm(64, 8)


@tvm.testing.requires_llvm
def test_llvm_bool():
    def check_llvm(n):
        A = te.placeholder((n,), name="A", dtype="int32")
        C = te.compute((n,), lambda i: A[i].equal(1).astype("float"), name="C")
        s = te.create_schedule(C.op)
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.randint(0, 2, size=(n,)).astype(A.dtype), ctx)
        c = tvm.nd.empty((n,), C.dtype, ctx)
        f(a, c)
        c_np = a.asnumpy() == 1
        tvm.testing.assert_allclose(c.asnumpy(), c_np)

    check_llvm(64)


@tvm.testing.requires_llvm
def test_rank_zero():
    def check_llvm(n):
        A = te.placeholder((n,), name="A")
        scale = te.placeholder((), name="scale")
        k = te.reduce_axis((0, n), name="k")
        C = te.compute((), lambda: te.sum(A[k] * scale(), axis=k), name="C")
        D = te.compute((), lambda: C() + 1)
        s = te.create_schedule(D.op)
        # build and invoke the kernel.
        f = tvm.build(s, [A, scale, D], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.randint(0, 2, size=(n,)).astype(A.dtype), ctx)
        sc = tvm.nd.array(np.random.randint(0, 2, size=()).astype(scale.dtype), ctx)
        d = tvm.nd.empty((), D.dtype, ctx)
        f(a, sc, d)
        d_np = np.sum(a.asnumpy()) * sc.asnumpy() + 1
        tvm.testing.assert_allclose(d.asnumpy(), d_np)

    check_llvm(64)


@tvm.testing.requires_llvm
def test_rank_zero_bound_checkers():
    def check_llvm(n):
        with tvm.transform.PassContext(config={"tir.instrument_bound_checkers": True}):
            A = te.placeholder((n,), name="A")
            scale = te.placeholder((), name="scale")
            k = te.reduce_axis((0, n), name="k")
            C = te.compute((), lambda: te.sum(A[k] * scale(), axis=k), name="C")
            D = te.compute((), lambda: C() + 1)
            s = te.create_schedule(D.op)
            # build and invoke the kernel.
            f = tvm.build(s, [A, scale, D], "llvm")
            ctx = tvm.cpu(0)
            # launch the kernel.
            a = tvm.nd.array(np.random.randint(0, 2, size=(n,)).astype(A.dtype), ctx)
            sc = tvm.nd.array(np.random.randint(0, 2, size=()).astype(scale.dtype), ctx)
            d = tvm.nd.empty((), D.dtype, ctx)
            f(a, sc, d)
            d_np = np.sum(a.asnumpy()) * sc.asnumpy() + 1
            tvm.testing.assert_allclose(d.asnumpy(), d_np)

    check_llvm(64)


@tvm.testing.requires_llvm
def test_alignment():
    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda i: A[i] * 3, name="B")
    s = te.create_schedule(B.op)
    bx, tx = s[B].split(B.op.axis[0], factor=8)
    s[B].vectorize(tx)
    f = tvm.build(s, [A, B], "llvm", name="test_alignment")

    lines = f.get_source().split("\n")

    # Check alignment on load/store.
    for l in lines:
        if "align" in l and "4 x float" in l:
            assert "align 32" in l

    # Check parameter alignment. This looks for the definition of the
    # outlined "compute_" function to see if there is an "align" attribute
    # listed there.
    def has_param_alignment():
        for l in lines:
            if re.search(r"test_alignment_compute_\([^(]*align [0-9]", l):
                return True
        return False

    if tvm.target.codegen.llvm_version_major() >= 5:
        assert has_param_alignment()

    # Check for assume intrinsics. This isn't 100% accurate, since it just
    # checks if the llvm.assume is there, but detailed check would require
    # a much more detailed analysis of the LLVM IR.
    def has_call_to_assume():
        for l in lines:
            if re.search(r"call.*llvm.assume", l):
                return True
        return False

    assert has_call_to_assume()


@tvm.testing.requires_llvm
def test_llvm_div():
    """Check that the semantics of div and mod is correct"""

    def check(start, end, dstart, dend, dtype, floor_div=False):
        div = tvm.te.floordiv if floor_div else tvm.tir.truncdiv
        mod = tvm.te.floormod if floor_div else tvm.tir.truncmod

        # A are dividends, B are divisors. Note that we add 1 to make include end in the range.
        A = te.placeholder((end - start + 1,), name="A", dtype=dtype)
        B = te.placeholder((dend - dstart + 1,), name="B", dtype=dtype)
        # We clip values with min and max so that simplifiers know the ranges of values

        def clipa(x):
            return tvm.te.min(tvm.tir.const(end, dtype), tvm.te.max(tvm.tir.const(start, dtype), x))

        def clipb(x):
            return tvm.te.min(
                tvm.tir.const(dend, dtype), tvm.te.max(tvm.tir.const(dstart, dtype), x)
            )

        # If the range is just a single point, use the constant itself
        if start == end:

            def clipa(x):
                return tvm.tir.const(start, dtype)

        if dstart == dend:

            def clipb(x):
                return tvm.tir.const(dstart, dtype)

        # D are division results and M are modulo results
        [D, M] = te.compute(
            (end - start + 1, dend - dstart + 1),
            lambda i, j: (div(clipa(A[i]), clipb(B[j])), mod(clipa(A[i]), clipb(B[j]))),
        )

        s = te.create_schedule([D.op, M.op])
        f = tvm.build(s, [A, B, D, M], "llvm")

        # Fill input arrays with values
        A_arr = tvm.nd.empty((end - start + 1,), dtype)
        B_arr = tvm.nd.empty((dend - dstart + 1,), dtype)
        A_arr.copyfrom(np.arange(start, end + 1, dtype=dtype))
        B_np = np.arange(dstart, dend + 1, dtype=dtype)
        # If the range of the divisor contains 0, replace it with 1 to avoid division by zero
        if dend >= 0 and dstart <= 0:
            B_np[-dstart] = 1
        B_arr.copyfrom(B_np)
        D_arr = tvm.nd.empty((end - start + 1, dend - dstart + 1), dtype)
        M_arr = tvm.nd.empty((end - start + 1, dend - dstart + 1), dtype)

        # Run the function and convert the results to numpy
        f(A_arr, B_arr, D_arr, M_arr)
        D_arr = D_arr.asnumpy()
        M_arr = M_arr.asnumpy()

        # This helper just prints additional info on failure
        def _show_info():
            print("dtype: {}".format(dtype))
            print("dividend range: [{}, {}]".format(start, end))
            print("divisor range: [{}, {}]".format(dstart, dend))
            lowered = tvm.lower(s, [A, B, D, M], simple_mode=True)
            print("Lowered code:")
            print(lowered)

        # Check that the computed values are correct
        for i in range(start, end + 1):
            for j in range(dstart, dend + 1):
                if j == 0:
                    continue

                if floor_div:
                    dref = i // j
                    mref = i % j
                else:
                    dref = int(float(i) / j)
                    mref = int(math.fmod(i, j))

                if D_arr[i - start, j - dstart] != dref:
                    _show_info()
                    raise AssertionError(
                        "Incorrect division result: {}({}, {}) is {} "
                        "but should be {}".format(
                            div.__name__, i, j, D_arr[i - start, j - dstart], dref
                        )
                    )
                if M_arr[i - start, j - dstart] != mref:
                    _show_info()
                    raise AssertionError(
                        "Incorrect modulo result: {}({}, {}) is {} "
                        "but should be {}".format(
                            mod.__name__, i, j, M_arr[i - start, j - dstart], mref
                        )
                    )

    # Try different ranges to cover different cases
    for start, end in [
        (-12, -12),
        (-11, -1),
        (-11, 0),
        (0, 0),
        (12, 12),
        (1, 11),
        (0, 11),
        (-11, 11),
    ]:
        for dstart, dend in [
            (-11, -1),
            (-11, 0),
            (-4, -4),
            (-2, -2),
            (1, 11),
            (0, 11),
            (4, 4),
            (2, 2),
            (-11, 11),
        ]:
            if end < start or dend < dstart or (dend == 0 and dstart == 0):
                continue
            check(start, end, dstart, dend, "int32", floor_div=False)
            check(start, end, dstart, dend, "int32", floor_div=True)
            check(start, end, dstart, dend, "int8", floor_div=False)
            check(start, end, dstart, dend, "int8", floor_div=True)
            if start >= 0 and dstart >= 0:
                check(start, end, dstart, dend, "uint32", floor_div=False)
                check(start, end, dstart, dend, "uint32", floor_div=True)

    # Additional tests for uint8
    for dstart, dend in [(0, 11), (1, 11), (2, 2), (4, 4)]:
        check(123, 133, dstart, dend, "uint8", floor_div=False)
        check(123, 133, dstart, dend, "uint8", floor_div=True)
        check(0, 255, dstart, dend, "uint8", floor_div=False)
        check(0, 255, dstart, dend, "uint8", floor_div=True)


@tvm.testing.requires_llvm
def test_llvm_fp_math():
    def check_llvm_reciprocal(n):
        A = te.placeholder((n,), name="A")
        B = te.compute((n,), lambda i: te.div(1.0, (1e37 * A[i])), name="B")

        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], "llvm")

        a = tvm.nd.array(np.full((n,), 100, "float32"))
        b = tvm.nd.empty((n,), "float32")
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), np.zeros((n,), "float32"))

    check_llvm_reciprocal(4)
    check_llvm_reciprocal(8)
    check_llvm_reciprocal(16)

    def check_llvm_sigmoid(n):
        A = te.placeholder((n,), name="A")
        B = te.compute((n,), lambda i: te.sigmoid(A[i]), name="B")

        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], "llvm")

        a = tvm.nd.array(np.full((n,), -1000, "float32"))
        b = tvm.nd.empty((n,), "float32")
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), np.zeros((n,), "float32"))

    check_llvm_sigmoid(4)
    check_llvm_sigmoid(8)
    check_llvm_sigmoid(16)


@tvm.testing.requires_llvm
def test_dwarf_debug_information():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = te.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    s[C].parallel(xo)
    s[C].vectorize(xi)

    def check_llvm_object():
        if tvm.target.codegen.llvm_version_major() < 5:
            return
        if tvm.target.codegen.llvm_version_major() > 6:
            return
        # build two functions
        f2 = tvm.lower(s, [A, B, C], name="fadd1")
        f1 = tvm.lower(s, [A, B, C], name="fadd2")
        m = tvm.build([f1, f2], "llvm")
        temp = utils.tempdir()
        o_path = temp.relpath("temp.o")
        m.save(o_path)
        import shutil
        import subprocess
        import sys

        # Try the dwarfdump utility (OS X)
        if shutil.which("dwarfdump"):
            output = subprocess.check_output(["dwarfdump", o_path])
            assert re.search(r"""DW_AT_name\\t\("fadd1"\)""", str(output))
            assert re.search(r"""DW_AT_name\\t\("fadd2"\)""", str(output))

        # Try gobjdump (OS X)
        if shutil.which("gobjdump"):
            output = subprocess.check_output(["gobjdump", "--dwarf", o_path])
            assert re.search(r"""DW_AT_name.*fadd1""", str(output))
            assert re.search(r"""DW_AT_name.*fadd2""", str(output))

        # Try objdump (Linux) - Darwin objdump has different DWARF syntax.
        if shutil.which("objdump") and sys.platform != "darwin":
            output = subprocess.check_output(["objdump", "--dwarf", o_path])
            assert re.search(r"""DW_AT_name.*fadd1""", str(output))
            assert re.search(r"""DW_AT_name.*fadd2""", str(output))

    def check_llvm_ir():
        if tvm.target.codegen.llvm_version_major() < 5:
            return
        if tvm.target.codegen.llvm_version_major() > 6:
            return
        # build two functions
        f2 = tvm.lower(s, [A, B, C], name="fadd1")
        f1 = tvm.lower(s, [A, B, C], name="fadd2")
        m = tvm.build([f1, f2], target="llvm -mtriple=aarch64-linux-gnu")
        ll = m.get_source("ll")

        # On non-Darwin OS, don't explicitly specify DWARF version.
        import re

        assert not re.search(r""""Dwarf Version""" "", ll)
        assert re.search(r"""llvm.dbg.value""", ll)

        # Try Darwin, require DWARF-2
        m = tvm.build([f1, f2], target="llvm -mtriple=x86_64-apple-darwin-macho")
        ll = m.get_source("ll")
        assert re.search(r"""i32 4, !"Dwarf Version", i32 2""", ll)
        assert re.search(r"""llvm.dbg.value""", ll)

    check_llvm_object()
    check_llvm_ir()


@tvm.testing.requires_llvm
def test_llvm_shuffle():
    a = te.placeholder((8,), "int32")
    b = te.placeholder((8,), "int32")
    c = te.compute((8,), lambda x: a[x] + b[7 - x])
    sch = te.create_schedule(c.op)

    def my_vectorize():
        def vectorizer(op):
            store = op.body
            idx = tvm.tir.Ramp(tvm.tir.const(0, "int32"), tvm.tir.const(1, "int32"), 8)
            all_ones = tvm.tir.const(1, "int32x8")
            value = store.value
            b_idx = tvm.tir.Shuffle([idx], [tvm.tir.const(i, "int32") for i in range(7, -1, -1)])
            new_a = tvm.tir.Load("int32x8", value.a.buffer_var, idx, all_ones)
            new_b = tvm.tir.Load("int32x8", value.b.buffer_var, b_idx, all_ones)
            value = new_a + new_b
            return tvm.tir.Store(store.buffer_var, new_a + new_b, idx, all_ones)

        def _transform(f, *_):
            return f.with_body(
                tvm.tir.stmt_functor.ir_transform(f.body, None, vectorizer, ["tir.For"])
            )

        return tvm.tir.transform.prim_func_pass(_transform, opt_level=0, name="my_vectorize")

    with tvm.transform.PassContext(config={"tir.add_lower_pass": [(1, my_vectorize())]}):
        ir = tvm.lower(sch, [a, b, c], simple_mode=True)
        module = tvm.build(sch, [a, b, c])
        a_ = tvm.nd.array(np.arange(1, 9, dtype="int32"))
        b_ = tvm.nd.array(np.arange(8, 0, -1, dtype="int32"))
        c_ = tvm.nd.array(np.zeros((8,), dtype="int32"))
        module(a_, b_, c_)
        tvm.testing.assert_allclose(c_.asnumpy(), (a_.asnumpy() * 2).astype("int32"))


def np_float2np_bf16(arr):
    """Convert a numpy array of float to a numpy array
    of bf16 in uint16"""
    orig = arr.view("<u4")
    bias = np.bitwise_and(np.right_shift(orig, 16), 1) + 0x7FFF
    return np.right_shift(orig + bias, 16).astype("uint16")


def np_float2tvm_bf16(arr):
    """Convert a numpy array of float to a TVM array
    of bf16"""
    nparr = np_float2np_bf16(arr)
    return tvm.nd.empty(nparr.shape, "uint16").copyfrom(nparr)


def np_bf162np_float(arr):
    """Convert a numpy array of bf16 (uint16) to a numpy array
    of float"""
    u32 = np.left_shift(arr.astype("uint32"), 16)
    return u32.view("<f4")


def np_bf16_cast_and_cast_back(arr):
    """ Convert a numpy array of float to bf16 and cast back"""
    return np_bf162np_float(np_float2np_bf16(arr))


@tvm.testing.requires_llvm
def test_llvm_bf16():
    def dotest(do_vectorize):
        np.random.seed(122)
        A = te.placeholder((32,), dtype="bfloat16")
        B = te.placeholder((32,), dtype="bfloat16")
        d = te.compute((32,), lambda x: A[x] + B[x])
        sch = te.create_schedule(d.op)
        print(tvm.lower(sch, [A, B, d]))
        if do_vectorize:
            sch[d].vectorize(d.op.axis[0])
        module = tvm.build(sch, [A, B, d])
        npa = np.random.rand(32).astype("float32")
        npb = np.random.rand(32).astype("float32")
        va = np_bf16_cast_and_cast_back(npa)
        vb = np_bf16_cast_and_cast_back(npb)
        res = np_bf16_cast_and_cast_back(va + vb)
        a_ = np_float2tvm_bf16(npa)
        b_ = np_float2tvm_bf16(npb)
        c_ = tvm.nd.empty((32,), "uint16")
        module(a_, b_, c_)
        tvm.testing.assert_allclose(np_bf162np_float(c_.asnumpy()), res)

    dotest(True)
    dotest(False)


@tvm.testing.requires_llvm
def test_llvm_crt_static_lib():
    A = te.placeholder((32,), dtype="bfloat16")
    B = te.placeholder((32,), dtype="bfloat16")
    d = te.compute((32,), lambda x: A[x] + B[x])
    sch = te.create_schedule(d.op)
    module = tvm.build(sch, [A, B, d], target=tvm.target.Target("llvm --system-lib --runtime=c"))
    print(module.get_source())
    module.save("test.o")


def atomic_add(x, y):
    return tvm.tir.call_intrin(y.dtype, "tir.atomic_add", x, y)


@tvm.testing.requires_llvm
def test_llvm_lower_atomic():
    def do_atomic_add(A):
        ib = tvm.tir.ir_builder.create()
        n = A.shape[0]
        atomic_add_return = ib.allocate(A.dtype, (1,), name="atomic_add_return", scope="local")
        one = tvm.tir.const(1, A.dtype)
        A_ptr = ib.buffer_ptr(A)
        with ib.for_range(0, n, name="i", kind="parallel") as i:
            atomic_add_return[0] = atomic_add(
                tvm.tir.call_intrin("handle", "tir.address_of", A_ptr[0]), one
            )
        return ib.get()

    A = tvm.te.placeholder((100,), dtype="int32", name="A")
    C = tvm.te.extern((100,), [A], lambda ins, _: do_atomic_add(ins[0]), name="C", dtype="int32")
    s = tvm.te.create_schedule(C.op)
    # This does not work because of pointer type mismatch
    # TVMError: LLVM module verification failed with the following errors:
    # Argument value type does not match pointer operand type!
    # %21 = atomicrmw add i8* %7, i32 1 monotonic
    # i8
    # f = tvm.build(s, [A], target="llvm")


@tvm.testing.requires_llvm
@tvm.testing.requires_gpu
def test_llvm_gpu_lower_atomic():
    def do_atomic_add(A):
        ib = tvm.tir.ir_builder.create()
        n = A.shape[0]
        atomic_add_return = ib.allocate(A.dtype, (1,), name="atomic_add_return", scope="local")
        one = tvm.tir.const(1, A.dtype)
        A_ptr = ib.buffer_ptr(A)
        nthread_tx = 64
        with ib.new_scope():
            nthread_bx = (n + nthread_tx - 1) // nthread_tx
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")
            ib.scope_attr(tx, "thread_extent", nthread_tx)
            ib.scope_attr(bx, "thread_extent", nthread_bx)
            atomic_add_return[0] = atomic_add(
                tvm.tir.call_intrin("handle", "tir.address_of", A_ptr[0]), one
            )
        return ib.get()

    size = 1024
    # CI uses LLVM 8, which does not support float atomic
    for dtype in ["int32"]:
        A = tvm.te.placeholder((size,), dtype=dtype, name="A")
        C = tvm.te.extern((size,), [A], lambda ins, _: do_atomic_add(ins[0]), dtype=dtype)
        s = tvm.te.create_schedule(C.op)
        f = tvm.build(s, [A], target="nvptx")

        ctx = tvm.gpu()
        a = tvm.nd.array(np.zeros((size,)).astype(A.dtype), ctx)
        f(a)
        ref = np.zeros((size,)).astype(A.dtype)
        ref[0] = size
        tvm.testing.assert_allclose(a.asnumpy(), ref, rtol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
