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
import topi
from tvm.contrib import util, clang
import numpy as np
import ctypes
import math

def test_llvm_intrin():
    ib = tvm.ir_builder.create()
    n = tvm.convert(4)
    A = ib.pointer("float32", name="A")
    args = [
        tvm.call_pure_intrin("handle", "tvm_address_of", A[0]),
        0, 3, 1
    ]
    ib.emit(tvm.make.Evaluate(
        tvm.make.Call(
            "int32", "prefetch", args, tvm.expr.Call.Intrinsic, None, 0)))
    body = ib.get()
    func = tvm.ir_pass.MakeAPI(body, "prefetch", [A], 0, True)
    fcode = tvm.build(func, None, "llvm")


def test_llvm_import():
    # extern "C" is necessary to get the correct signature
    cc_code = """
    extern "C" float my_add(float x, float y) {
      return x + y;
    }
    """
    n = 10
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute((n,), lambda *i:
                    tvm.call_pure_extern("float32", "my_add", A(*i), 1.0),
                    name='B')
    def check_llvm(use_file):
        if not tvm.module.enabled("llvm"):
            return
        if not clang.find_clang(required=False):
            print("skip because clang is not available")
            return
        temp = util.tempdir()
        ll_path = temp.relpath("temp.ll")
        ll_code = clang.create_llvm(cc_code, output=ll_path)
        s = tvm.create_schedule(B.op)
        if use_file:
            s[B].pragma(s[B].op.axis[0], "import_llvm", ll_path)
        else:
            s[B].pragma(s[B].op.axis[0], "import_llvm", ll_code)
        # BUILD and invoke the kernel.
        f = tvm.build(s, [A, B], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        f(a, b)
        tvm.testing.assert_allclose(
            b.asnumpy(), a.asnumpy() + 1.0)
    check_llvm(use_file=True)
    check_llvm(use_file=False)



def test_llvm_lookup_intrin():
    ib = tvm.ir_builder.create()
    m = tvm.var("m")
    A = ib.pointer("uint8x8", name="A")
    x = tvm.call_llvm_intrin("uint8x8", "llvm.ctpop.i8", tvm.const(1, 'uint32'), A)
    ib.emit(x)
    body = ib.get()
    func = tvm.ir_pass.MakeAPI(body, "ctpop", [A], 1, True)
    fcode = tvm.build(func, None, "llvm")


def test_llvm_add_pipeline():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    AA = tvm.compute((n,), lambda *i: A(*i), name='A')
    BB = tvm.compute((n,), lambda *i: B(*i), name='B')
    T = tvm.compute(A.shape, lambda *i: AA(*i) + BB(*i), name='T')
    C = tvm.compute(A.shape, lambda *i: T(*i), name='C')
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    xo1, xo2 = s[C].split(xo, factor=13)
    s[C].parallel(xo2)
    s[C].pragma(xo1, "parallel_launch_point")
    s[C].pragma(xo2, "parallel_stride_pattern")
    s[C].pragma(xo2, "parallel_barrier_when_finish")
    s[C].vectorize(xi)

    def check_llvm():
        if not tvm.module.enabled("llvm"):
            return
        # Specifically allow offset to test codepath when offset is available
        Ab = tvm.decl_buffer(
            A.shape, A.dtype,
            elem_offset=tvm.var('Aoffset'),
            offset_factor=8,
            name='A')
        binds = {A : Ab}
        # BUILD and invoke the kernel.
        f = tvm.build(s, [A, B, C], "llvm", binds=binds)
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        f(a, b, c)
        tvm.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())

    with tvm.build_config(offset_factor=4):
        check_llvm()


def test_llvm_persist_parallel():
    n = 128
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1, name='B')
    C = tvm.compute(A.shape, lambda *i: tvm.sqrt(B(*i)) * 2 + 2, name='C')
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=8)
    xo1, xo2 = s[C].split(xo, nparts=1)
    s[B].compute_at(s[C], xo1)
    s[B].parallel(s[B].op.axis[0])
    s[B].pragma(s[B].op.axis[0], "parallel_barrier_when_finish")
    s[C].parallel(xi)
    s[C].pragma(xo1, "parallel_launch_point")
    s[C].pragma(xi, "parallel_stride_pattern")

    def check_llvm():
        if not tvm.module.enabled("llvm"):
            return
        # BUILD and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        f(a, c)
        tvm.testing.assert_allclose(c.asnumpy(),
                                   np.sqrt(a.asnumpy() + 1) * 2 + 2,
                                   rtol=1e-5)

    check_llvm()


def test_llvm_flip_pipeline():
    def check_llvm(nn, base):
        if not tvm.module.enabled("llvm"):
            return
        n = tvm.convert(nn)
        A = tvm.placeholder((n + base), name='A')
        C = tvm.compute((n,), lambda i: A(nn + base- i - 1), name='C')
        s = tvm.create_schedule(C.op)
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
        tvm.testing.assert_allclose(
            c.asnumpy(), a.asnumpy()[::-1][:n])
    check_llvm(4, 0)
    check_llvm(128, 8)
    check_llvm(3, 0)
    check_llvm(128, 1)


def test_llvm_vadd_pipeline():
    def check_llvm(n, lanes):
        if not tvm.module.enabled("llvm"):
            return
        A = tvm.placeholder((n,), name='A', dtype="float32x%d" % lanes)
        B = tvm.compute((n,), lambda i: A[i], name='B')
        C = tvm.compute((n,), lambda i: B[i] + tvm.const(1, A.dtype), name='C')
        s = tvm.create_schedule(C.op)
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
        a = tvm.nd.empty((n,), A.dtype).copyfrom(
            np.random.uniform(size=(n, lanes)))
        c = tvm.nd.empty((n,), C.dtype, ctx)
        f(a, c)
        tvm.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + 1)
    check_llvm(64, 2)
    check_llvm(512, 2)


def test_llvm_madd_pipeline():
    def check_llvm(nn, base, stride):
        if not tvm.module.enabled("llvm"):
            return
        n = tvm.convert(nn)
        A = tvm.placeholder((n + base, stride), name='A')
        C = tvm.compute((n, stride), lambda i, j: A(base + i, j) + 1, name='C')
        s = tvm.create_schedule(C.op)
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
        tvm.testing.assert_allclose(
            c.asnumpy(), a.asnumpy()[base:] + 1)
    check_llvm(64, 0, 2)
    check_llvm(4, 0, 1)
    with tvm.build_config(restricted_func=False):
        check_llvm(4, 0, 3)


def test_llvm_temp_space():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda i: A(i) + 1, name='B')
    C = tvm.compute(A.shape, lambda i: B(i) + 1, name='C')
    s = tvm.create_schedule(C.op)

    def check_llvm():
        if not tvm.module.enabled("llvm"):
            return
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        f(a, c)
        tvm.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + 1 + 1)
    check_llvm()

def test_multiple_func():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    s[C].parallel(xo)
    s[C].vectorize(xi)
    def check_llvm():
        if not tvm.module.enabled("llvm"):
            return
        # build two functions
        f2 = tvm.lower(s, [A, B, C], name="fadd1")
        f1 = tvm.lower(s, [A, B, C], name="fadd2")
        m = tvm.build([f1, f2], "llvm")
        fadd1 = m['fadd1']
        fadd2 = m['fadd2']
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        fadd1(a, b, c)
        tvm.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())
        fadd2(a, b, c)
        tvm.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())
    check_llvm()



def test_llvm_condition():
    def check_llvm(n, offset):
        if not tvm.module.enabled("llvm"):
            return
        A = tvm.placeholder((n, ), name='A')
        C = tvm.compute((n,), lambda i: tvm.if_then_else(i >= offset, A[i], 0.0), name='C')
        s = tvm.create_schedule(C.op)
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


def test_llvm_bool():
    def check_llvm(n):
        if not tvm.module.enabled("llvm"):
            return
        A = tvm.placeholder((n, ), name='A', dtype="int32")
        C = tvm.compute((n,), lambda i: A[i].equal(1).astype("float"), name='C')
        s = tvm.create_schedule(C.op)
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


def test_rank_zero():
    def check_llvm(n):
        if not tvm.module.enabled("llvm"):
            return
        A = tvm.placeholder((n, ), name='A')
        scale = tvm.placeholder((), name='scale')
        k = tvm.reduce_axis((0, n), name="k")
        C = tvm.compute((), lambda : tvm.sum(A[k] * scale(), axis=k), name="C")
        D = tvm.compute((), lambda : C() + 1)
        s = tvm.create_schedule(D.op)
        # build and invoke the kernel.
        f = tvm.build(s, [A, scale, D], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.randint(0, 2, size=(n,)).astype(A.dtype), ctx)
        sc = tvm.nd.array(
            np.random.randint(0, 2, size=()).astype(scale.dtype), ctx)
        d = tvm.nd.empty((), D.dtype, ctx)
        f(a, sc, d)
        d_np = np.sum(a.asnumpy()) * sc.asnumpy() + 1
        tvm.testing.assert_allclose(d.asnumpy(), d_np)
    check_llvm(64)

def test_rank_zero_bound_checkers():
    def check_llvm(n):
        if not tvm.module.enabled("llvm"):
            return
        with tvm.build_config(instrument_bound_checkers=True):
            A = tvm.placeholder((n, ), name='A')
            scale = tvm.placeholder((), name='scale')
            k = tvm.reduce_axis((0, n), name="k")
            C = tvm.compute((), lambda : tvm.sum(A[k] * scale(), axis=k), name="C")
            D = tvm.compute((), lambda : C() + 1)
            s = tvm.create_schedule(D.op)
            # build and invoke the kernel.
            f = tvm.build(s, [A, scale, D], "llvm")
            ctx = tvm.cpu(0)
            # launch the kernel.
            a = tvm.nd.array(np.random.randint(0, 2, size=(n,)).astype(A.dtype), ctx)
            sc = tvm.nd.array(
                np.random.randint(0, 2, size=()).astype(scale.dtype), ctx)
            d = tvm.nd.empty((), D.dtype, ctx)
            f(a, sc, d)
            d_np = np.sum(a.asnumpy()) * sc.asnumpy() + 1
            tvm.testing.assert_allclose(d.asnumpy(), d_np)
    check_llvm(64)


def test_alignment():
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda i: A[i] * 3, name='B')
    s = tvm.create_schedule(B.op)
    bx, tx = s[B].split(B.op.axis[0], factor=8)
    s[B].vectorize(tx)
    f = tvm.build(s, [A, B], "llvm")

    for l in f.get_source().split("\n"):
        if "align" in l and "4 x float" in l:
            assert "align 32" in l

def test_llvm_div():
    """Check that the semantics of div and mod is the same as in C/C++"""
    def check_div(start, end, divisor, dtype):
        T = tvm.compute((end - start,),
                        lambda i: tvm.div(tvm.expr.Cast(dtype, (start + i)), tvm.const(divisor, dtype)))
        s = tvm.create_schedule([T.op])
        f = tvm.build(s, [T], "llvm")
        a = tvm.nd.empty((end - start,), dtype)
        f(a)
        ref = [int(float(i)/divisor) for i in range(start, end)]
        tvm.testing.assert_allclose(a.asnumpy(), ref)

    def check_mod(start, end, divisor, dtype):
        tmod = tvm.truncmod
        T = tvm.compute((end - start,),
                        lambda i: tmod(tvm.expr.Cast(dtype, (start + i)), tvm.const(divisor, dtype)))
        s = tvm.create_schedule([T.op])
        f = tvm.build(s, [T], "llvm")
        a = tvm.nd.empty((end - start,), dtype)
        f(a)
        ref = [int(math.fmod(i, divisor)) for i in range(start, end)]
        tvm.testing.assert_allclose(a.asnumpy(), ref)

    def check_llvm(start, end, divisor, dtype):
        check_div(start, end, divisor, dtype)
        check_mod(start, end, divisor, dtype)

    for d in range(-5, 6):
        if d != 0:
            # Note that 11 (and not e.g. 10) is used to avoid issues with the simplifier
            check_llvm(-11, 11, d, 'int32')
            check_llvm(-11, 11, d, 'int8')
            if d > 0:
                check_llvm(123, 133, d, 'uint8')
                check_llvm(0, 256, d, 'uint8')

def test_llvm_fp_math():
    def check_llvm_reciprocal(n):
        A = tvm.placeholder((n,), name='A')
        B = tvm.compute((n,), lambda i: tvm.div(1.0,(1e+37*A[i])), name='B')

        s = tvm.create_schedule(B.op)
        f = tvm.build(s, [A, B], "llvm")

        a = tvm.nd.array(np.full((n,), 100, 'float32'))
        b = tvm.nd.empty((n,), 'float32')
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), np.zeros((n,), 'float32'))

    check_llvm_reciprocal(4)
    check_llvm_reciprocal(8)
    check_llvm_reciprocal(16)

    def check_llvm_sigmoid(n):
        A = tvm.placeholder((n,), name='A')
        B = tvm.compute((n,), lambda i: tvm.sigmoid(A[i]), name='B')

        s = tvm.create_schedule(B.op)
        f = tvm.build(s, [A, B], "llvm")

        a = tvm.nd.array(np.full((n,), -1000, 'float32'))
        b = tvm.nd.empty((n,), 'float32')
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), np.zeros((n,), 'float32'))

    check_llvm_sigmoid(4)
    check_llvm_sigmoid(8)
    check_llvm_sigmoid(16)


def test_dwarf_debug_information():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    s[C].parallel(xo)
    s[C].vectorize(xi)
    def check_llvm_object():
        if not tvm.module.enabled("llvm"):
            return
        if tvm.codegen.llvm_version_major() < 5:
            return
        if tvm.codegen.llvm_version_major() > 6:
            return
        # build two functions
        f2 = tvm.lower(s, [A, B, C], name="fadd1")
        f1 = tvm.lower(s, [A, B, C], name="fadd2")
        m = tvm.build([f1, f2], "llvm")
        temp = util.tempdir()
        o_path = temp.relpath("temp.o")
        m.save(o_path)
        import re
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
        if shutil.which("objdump") and sys.platform != 'darwin':
            output = subprocess.check_output(["objdump", "--dwarf", o_path])
            assert re.search(r"""DW_AT_name.*fadd1""", str(output))
            assert re.search(r"""DW_AT_name.*fadd2""", str(output))

    def check_llvm_ir():
        if not tvm.module.enabled("llvm"):
            return
        if tvm.codegen.llvm_version_major() < 5:
            return
        if tvm.codegen.llvm_version_major() > 6:
            return
        # build two functions
        f2 = tvm.lower(s, [A, B, C], name="fadd1")
        f1 = tvm.lower(s, [A, B, C], name="fadd2")
        m = tvm.build([f1, f2], target="llvm -target=aarch64-linux-gnu")
        ll = m.get_source("ll")

        # On non-Darwin OS, don't explicitly specify DWARF version.
        import re
        assert not re.search(r""""Dwarf Version""""", ll)
        assert re.search(r"""llvm.dbg.value""", ll)

        # Try Darwin, require DWARF-2
        m = tvm.build([f1, f2],
                      target="llvm -target=x86_64-apple-darwin-macho")
        ll = m.get_source("ll")
        assert re.search(r"""i32 4, !"Dwarf Version", i32 2""", ll)
        assert re.search(r"""llvm.dbg.value""", ll)

    check_llvm_object()
    check_llvm_ir()


def test_llvm_shuffle():
    a = tvm.placeholder((8, ), 'int32')
    b = tvm.placeholder((8, ), 'int32')
    c = tvm.compute((8, ), lambda x: a[x] + b[7-x])
    sch = tvm.create_schedule(c.op)

    def my_vectorize(stmt):

        def vectorizer(op):
            store = op.body
            idx = tvm.make.Ramp(tvm.const(0, 'int32'), tvm.const(1, 'int32'), 8)
            all_ones = tvm.const(1, 'int32x8')
            value = store.value
            b_idx = tvm.make.Shuffle([idx], [tvm.const(i, 'int32') for i in range(7, -1, -1)])
            new_a = tvm.make.Load('int32x8', value.a.buffer_var, idx, all_ones)
            new_b = tvm.make.Load('int32x8', value.b.buffer_var, b_idx, all_ones)
            value = new_a + new_b
            return tvm.make.Store(store.buffer_var, new_a + new_b, idx, all_ones)

        return tvm.ir_pass.IRTransform(stmt, None, vectorizer, ['For'])

    with tvm.build_config(add_lower_pass=[(1, my_vectorize)]):
        ir = tvm.lower(sch, [a, b, c], simple_mode=True)
        module = tvm.build(sch, [a, b, c])
        a_ = tvm.ndarray.array(np.arange(1, 9, dtype='int32'))
        b_ = tvm.ndarray.array(np.arange(8, 0, -1, dtype='int32'))
        c_ = tvm.ndarray.array(np.zeros((8, ), dtype='int32'))
        module(a_, b_, c_)
        tvm.testing.assert_allclose(c_.asnumpy(), (a_.asnumpy() * 2).astype('int32'))

if __name__ == "__main__":
    test_llvm_import()
    test_alignment()
    test_rank_zero()
    test_rank_zero_bound_checkers()
    test_llvm_bool()
    test_llvm_persist_parallel()
    test_llvm_condition()
    test_llvm_vadd_pipeline()
    test_llvm_add_pipeline()
    test_llvm_intrin()
    test_multiple_func()
    test_llvm_flip_pipeline()
    test_llvm_madd_pipeline()
    test_llvm_temp_space()
    test_llvm_lookup_intrin()
    test_llvm_div()
    test_llvm_fp_math()
    test_dwarf_debug_information()
    test_llvm_shuffle()
