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
import math
import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.contrib import clang, utils
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target.codegen import llvm_get_intrinsic_name, llvm_lookup_intrinsic_id


@tvm.testing.requires_llvm
def test_llvm_intrin():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.handle("float32")):
            A_buf = T.Buffer((4,), "float32", data=A)
            T.evaluate(T.Call("void", "tir.prefetch", [T.address_of(A_buf[0]), 0, 3, 1]))

    fcode = tvm.compile(Module)


@tvm.testing.requires_llvm
def test_llvm_void_intrin():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.handle("uint8")):
            # Create an intrinsic that returns void.
            T.call_llvm_intrin("", "llvm.assume", T.bool(True))

    fcode = tvm.compile(Module)


@tvm.testing.requires_llvm
def test_llvm_intrinsic_id():
    orig_name = "llvm.x86.sse2.pmadd.wd"
    intrin_id = llvm_lookup_intrinsic_id(orig_name)
    name = llvm_get_intrinsic_name(intrin_id)
    assert orig_name == name


@tvm.testing.requires_llvm
def test_llvm_overloaded_intrin():
    # Name lookup for overloaded intrinsics in LLVM 4- requires a name
    # that includes the overloaded types.
    if tvm.target.codegen.llvm_version_major() < 5:
        return

    # int1 is the type for the is_zero_undef parameter
    int1_zero = tvm.tir.const(0, "int1")

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((1, 1), "int32"), C: T.Buffer((1, 1), "int32")):
            with T.sblock("C"):
                T.reads()
                T.writes()
                C[0, 0] = T.call_llvm_pure_intrin("int32", "llvm.ctlz", A[0, 0], int1_zero)

    f = tvm.compile(Module, target="llvm")


@tvm.testing.requires_llvm
def test_llvm_lookup_intrin():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.handle("uint8x8")):
            A_buf = T.Buffer((1,), "uint8x8", data=A)
            T.evaluate(T.call_llvm_pure_intrin("uint8x8", "llvm.ctpop.v8i8", T.uint32(1), A_buf[0]))

    fcode = tvm.compile(Module, None)


@tvm.testing.requires_llvm
def test_llvm_large_uintimm():
    value = (1 << 63) + 123
    large_val = tvm.tir.const(value, "uint64")

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((), "uint64")):
            T.func_attr({"tir.noalias": True})
            with T.sblock("A"):
                vi = T.axis.spatial(1, 0)
                T.reads()
                T.writes(A[()])
                A[()] = large_val + T.uint64(3)

    f = tvm.compile(Module, target="llvm")
    dev = tvm.cpu(0)
    a = tvm.runtime.empty((), dtype="uint64", device=dev)
    f(a)
    assert a.numpy() == value + 3


@tvm.testing.requires_llvm
def test_llvm_multi_parallel():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((128,), "float32"), C: T.Buffer((128,), "float32")):
            T.func_attr({"tir.noalias": True})
            B = T.alloc_buffer((128,))
            for i0_0_0 in T.parallel(1):
                for ax0 in range(128):
                    with T.sblock("B"):
                        v_i0 = T.axis.spatial(128, ax0)
                        T.reads(A[v_i0])
                        T.writes(B[v_i0])
                        B[v_i0] = A[v_i0] + T.float32(1.0)
                for i0_0_1 in range(16):
                    for i0_1 in T.parallel(8):
                        with T.sblock("C"):
                            v_i0 = T.axis.spatial(128, i0_0_0 * 128 + i0_0_1 * 8 + i0_1)
                            T.reads(B[v_i0])
                            T.writes(C[v_i0])
                            C[v_i0] = T.sqrt(B[v_i0]) * T.float32(2.0) + T.float32(2.0)

    n = 128
    f = tvm.compile(Module, target="llvm")
    dev = tvm.cpu(0)
    a = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
    c = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
    f(a, c)
    tvm.testing.assert_allclose(c.numpy(), np.sqrt(a.numpy() + 1) * 2 + 2, rtol=1e-5)


@tvm.testing.requires_llvm
def test_llvm_flip_pipeline():
    def check_llvm(nn, base):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((nn + base,), "float32"), C: T.Buffer((nn,), "float32")):
                T.func_attr({"tir.noalias": True})
                for i_0 in T.parallel((nn + 3) // 4):
                    for i_1 in T.vectorized(4):
                        with T.sblock("C"):
                            v_i = T.axis.spatial(nn, i_0 * 4 + i_1)
                            T.where(i_0 * 4 + i_1 < nn)
                            T.reads(A[nn + base - 1 - v_i])
                            T.writes(C[v_i])
                            C[v_i] = A[nn + base - 1 - v_i]

        f = tvm.compile(Module, target="llvm")
        dev = tvm.cpu(0)
        a = tvm.runtime.tensor(np.random.uniform(size=(nn + base)).astype("float32"), dev)
        c = tvm.runtime.tensor(np.zeros(nn, dtype="float32"), dev)
        f(a, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy()[::-1][:nn])

    check_llvm(4, 0)
    check_llvm(128, 8)
    check_llvm(3, 0)
    check_llvm(128, 1)


@tvm.testing.requires_llvm
def test_llvm_vadd_pipeline():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            n = T.int32(is_size_var=True)
            A = T.match_buffer(var_A, (n,))
            B = T.match_buffer(var_B, (n,))
            C = T.match_buffer(var_C, (n,))
            for i_0 in range((n + 3) // 4):
                for i_1 in T.vectorized(4):
                    with T.sblock("C"):
                        v_i = T.axis.spatial(n, i_0 * 4 + i_1)
                        T.where(i_0 * 4 + i_1 < n)
                        T.reads(A[v_i], B[v_i])
                        T.writes(C[v_i])
                        C[v_i] = A[v_i] + B[v_i]

    f = tvm.compile(Module, target="llvm")
    dev = tvm.cpu(0)
    n = 128
    a = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
    b = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
    c = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


@tvm.testing.requires_llvm
def test_llvm_madd_pipeline():
    def check_llvm(nn, base, stride):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(
                A: T.Buffer((nn + base, stride), "float32"),
                C: T.Buffer((nn, stride), "float32"),
            ):
                T.func_attr({"tir.noalias": True})
                for i_0 in T.parallel((nn + 3) // 4):
                    for i_1 in T.vectorized(4):
                        for j in range(stride):
                            with T.sblock("C"):
                                v_i = T.axis.spatial(nn, i_0 * 4 + i_1)
                                v_j = T.axis.spatial(stride, j)
                                T.where(i_0 * 4 + i_1 < nn)
                                T.reads(A[v_i + base, v_j])
                                T.writes(C[v_i, v_j])
                                C[v_i, v_j] = A[v_i + base, v_j] + T.float32(1.0)

        f = tvm.compile(Module, target="llvm")
        dev = tvm.cpu(0)
        a = tvm.runtime.tensor(np.random.uniform(size=(nn + base, stride)).astype("float32"), dev)
        c = tvm.runtime.tensor(np.zeros((nn, stride), dtype="float32"), dev)
        f(a, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy()[base:] + 1)

    check_llvm(64, 0, 2)
    check_llvm(4, 0, 1)

    with tvm.transform.PassContext(config={"tir.noalias": False}):
        check_llvm(4, 0, 3)


@tvm.testing.requires_llvm
def test_llvm_temp_space():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((1024,), "float32"), C: T.Buffer((1024,), "float32")):
            T.func_attr({"tir.noalias": True})
            B = T.alloc_buffer((1024,))
            for i in range(1024):
                with T.sblock("B"):
                    v_i = T.axis.spatial(1024, i)
                    T.reads(A[v_i])
                    T.writes(B[v_i])
                    B[v_i] = A[v_i] + T.float32(1.0)
            for i in range(1024):
                with T.sblock("C"):
                    v_i = T.axis.spatial(1024, i)
                    T.reads(B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = B[v_i] + T.float32(1.0)

    nn = 1024
    f = tvm.compile(Module, target="llvm")
    dev = tvm.cpu(0)
    a = tvm.runtime.tensor(np.random.uniform(size=nn).astype("float32"), dev)
    c = tvm.runtime.tensor(np.zeros(nn, dtype="float32"), dev)
    f(a, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + 1 + 1)


@tvm.testing.requires_llvm
def test_multiple_func():
    @I.ir_module
    class Module:
        @T.prim_func
        def fadd1(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            n = T.int32(is_size_var=True)
            A = T.match_buffer(var_A, (n,))
            B = T.match_buffer(var_B, (n,))
            C = T.match_buffer(var_C, (n,))
            for i in range(n):
                with T.sblock("C"):
                    v_i = T.axis.spatial(n, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] + B[v_i]

        @T.prim_func
        def fadd2(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            n = T.int32(is_size_var=True)
            A = T.match_buffer(var_A, (n,))
            B = T.match_buffer(var_B, (n,))
            C = T.match_buffer(var_C, (n,))
            for i in range(n):
                with T.sblock("C"):
                    v_i = T.axis.spatial(n, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] + B[v_i]

    f = tvm.compile(Module, target="llvm")
    dev = tvm.cpu(0)
    n = 10
    a = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
    b = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
    c = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)

    f["fadd1"](a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
    f["fadd2"](a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


@tvm.testing.requires_llvm
def test_llvm_condition():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((64,), "float32"), C: T.Buffer((64,), "float32")):
            T.func_attr({"tir.noalias": True})
            for i in range(64):
                with T.sblock("C"):
                    v_i = T.axis.spatial(64, i)
                    T.reads(A[v_i])
                    T.writes(C[v_i])
                    C[v_i] = T.if_then_else(8 <= v_i, A[v_i], T.float32(0.0))

    n = 64
    offset = 8
    f = tvm.compile(Module, target="llvm")
    dev = tvm.cpu(0)
    a = tvm.runtime.tensor(np.random.uniform(size=(n,)).astype("float32"), dev)
    c = tvm.runtime.empty((n,), "float32", dev)
    f(a, c)
    c_np = a.numpy()
    c_np[:offset] = 0
    tvm.testing.assert_allclose(c.numpy(), c_np)


@tvm.testing.requires_llvm
def test_llvm_bool():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((64,), "int32"), C: T.Buffer((64,), "float32")):
            T.func_attr({"tir.noalias": True})
            for i in range(64):
                with T.sblock("C"):
                    v_i = T.axis.spatial(64, i)
                    T.reads(A[v_i])
                    T.writes(C[v_i])
                    C[v_i] = T.Cast("float32", A[v_i] == 1)

    n = 64
    f = tvm.compile(Module, target="llvm")
    dev = tvm.cpu(0)
    a = tvm.runtime.tensor(np.random.randint(0, 2, size=(n,)).astype("int32"), dev)
    c = tvm.runtime.empty((n,), "float32", dev)
    f(a, c)
    c_np = a.numpy() == 1
    tvm.testing.assert_allclose(c.numpy(), c_np)


@tvm.testing.requires_llvm
def test_llvm_cast_float_to_bool():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "bool")):
            T.func_attr({"tir.noalias": True})
            for i in range(4):
                with T.sblock("C"):
                    v_i = T.axis.spatial(4, i)
                    T.reads(A[v_i])
                    T.writes(C[v_i])
                    C[v_i] = T.Cast("bool", A[v_i])

    n = 4
    f = tvm.compile(Module, target="llvm")
    dev = tvm.cpu(0)
    a = tvm.runtime.tensor(np.array([0.0, 1.0, np.nan, np.inf], dtype="float32"), dev)
    c = tvm.runtime.empty((n,), dtype="bool", device=dev)
    f(a, c)
    c_np = np.array([False, True, True, True], dtype="bool")
    tvm.testing.assert_allclose(c.numpy(), c_np)


@tvm.testing.requires_llvm
def test_rank_zero():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(
            A: T.Buffer((64,), "float32"),
            scale: T.Buffer((), "float32"),
            compute: T.Buffer((), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            C = T.alloc_buffer(())
            for k in range(64):
                with T.sblock("C"):
                    v_k = T.axis.reduce(64, k)
                    T.reads(A[v_k], scale[()])
                    T.writes(C[()])
                    with T.init():
                        C[()] = T.float32(0.0)
                    C[()] = C[()] + A[v_k] * scale[()]
            with T.sblock("compute"):
                vi = T.axis.spatial(1, 0)
                T.reads(C[()])
                T.writes(compute[()])
                compute[()] = C[()] + T.float32(1.0)

    n = 64
    f = tvm.compile(Module, target="llvm")
    dev = tvm.cpu(0)
    a = tvm.runtime.tensor(np.random.randint(0, 2, size=(n,)).astype("float32"), dev)
    sc = tvm.runtime.tensor(np.random.randint(0, 2, size=()).astype("float32"), dev)
    d = tvm.runtime.empty((), "float32", dev)
    f(a, sc, d)
    d_np = np.sum(a.numpy()) * sc.numpy() + 1
    tvm.testing.assert_allclose(d.numpy(), d_np)


@tvm.testing.requires_llvm
def test_rank_zero_bound_checkers():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(
            A: T.Buffer((64,), "float32"),
            scale: T.Buffer((), "float32"),
            compute: T.Buffer((), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            C = T.alloc_buffer(())
            for k in range(64):
                with T.sblock("C"):
                    v_k = T.axis.reduce(64, k)
                    T.reads(A[v_k], scale[()])
                    T.writes(C[()])
                    with T.init():
                        C[()] = T.float32(0.0)
                    C[()] = C[()] + A[v_k] * scale[()]
            with T.sblock("compute"):
                vi = T.axis.spatial(1, 0)
                T.reads(C[()])
                T.writes(compute[()])
                compute[()] = C[()] + T.float32(1.0)

    n = 64
    with tvm.transform.PassContext(config={"tir.instrument_bound_checkers": True}):
        f = tvm.compile(Module, target="llvm")
        dev = tvm.cpu(0)
        a = tvm.runtime.tensor(np.random.randint(0, 2, size=(n,)).astype("float32"), dev)
        sc = tvm.runtime.tensor(np.random.randint(0, 2, size=()).astype("float32"), dev)
        d = tvm.runtime.empty((), "float32", dev)
        f(a, sc, d)
        d_np = np.sum(a.numpy()) * sc.numpy() + 1
        tvm.testing.assert_allclose(d.numpy(), d_np)


@tvm.testing.requires_llvm
def test_alignment():
    @I.ir_module
    class Module:
        @T.prim_func
        def test_alignment(A: T.Buffer((1024,), "float32"), B: T.Buffer((1024,), "float32")):
            T.func_attr({"tir.noalias": True})
            for i_0 in range(128):
                for i_1 in T.vectorized(8):
                    with T.sblock("B"):
                        v_i = T.axis.spatial(1024, i_0 * 8 + i_1)
                        T.reads(A[v_i])
                        T.writes(B[v_i])
                        B[v_i] = A[v_i] * T.float32(3.0)

    f = tvm.tir.build(Module, target="llvm")

    lines = f.inspect_source().split("\n")

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
        a_size = end - start + 1
        b_size = dend - dstart + 1

        div_fn = tvm.tir.floordiv if floor_div else tvm.tir.truncdiv
        mod_fn = tvm.tir.floormod if floor_div else tvm.tir.truncmod

        # Build clipping helpers — capture TIR const values from env
        _start = tvm.tir.const(start, dtype)
        _end = tvm.tir.const(end, dtype)
        _dstart = tvm.tir.const(dstart, dtype)
        _dend = tvm.tir.const(dend, dtype)

        if start == end:
            clipa = lambda x: _start
        else:
            clipa = lambda x: T.min(_end, T.max(_start, x))

        if dstart == dend:
            clipb = lambda x: _dstart
        else:
            clipb = lambda x: T.min(_dend, T.max(_dstart, x))

        @I.ir_module
        class Module:
            @T.prim_func
            def main(
                A: T.Buffer((a_size,), dtype),
                B: T.Buffer((b_size,), dtype),
                D: T.Buffer((a_size, b_size), dtype),
                M: T.Buffer((a_size, b_size), dtype),
            ):
                T.func_attr({"tir.noalias": True})
                for i, j in T.grid(a_size, b_size):
                    with T.sblock("D"):
                        v_i, v_j = T.axis.remap("SS", [i, j])
                        T.reads(A[v_i], B[v_j])
                        T.writes(D[v_i, v_j])
                        D[v_i, v_j] = div_fn(clipa(A[v_i]), clipb(B[v_j]))
                    with T.sblock("M"):
                        v_i, v_j = T.axis.remap("SS", [i, j])
                        T.reads(A[v_i], B[v_j])
                        T.writes(M[v_i, v_j])
                        M[v_i, v_j] = mod_fn(clipa(A[v_i]), clipb(B[v_j]))

        f = tvm.compile(Module, target="llvm")

        # Fill input arrays with values
        A_arr = tvm.runtime.empty((a_size,), dtype)
        B_arr = tvm.runtime.empty((b_size,), dtype)
        A_arr.copyfrom(np.arange(start, end + 1, dtype=dtype))
        B_np = np.arange(dstart, dend + 1, dtype=dtype)
        # If the range of the divisor contains 0, replace it with 1 to avoid division by zero
        if dend >= 0 and dstart <= 0:
            B_np[-dstart] = 1
        B_arr.copyfrom(B_np)
        D_arr = tvm.runtime.empty((a_size, b_size), dtype)
        M_arr = tvm.runtime.empty((a_size, b_size), dtype)

        # Run the function and convert the results to numpy
        f(A_arr, B_arr, D_arr, M_arr)
        D_arr = D_arr.numpy()
        M_arr = M_arr.numpy()

        # This helper just prints additional info on failure
        def _show_info():
            print(f"dtype: {dtype}")
            print(f"dividend range: [{start}, {end}]")
            print(f"divisor range: [{dstart}, {dend}]")

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
                        f"Incorrect division result: {div_fn.__name__}({i}, {j}) is {D_arr[i - start, j - dstart]} "
                        f"but should be {dref}"
                    )
                if M_arr[i - start, j - dstart] != mref:
                    _show_info()
                    raise AssertionError(
                        f"Incorrect modulo result: {mod_fn.__name__}({i}, {j}) is {M_arr[i - start, j - dstart]} "
                        f"but should be {mref}"
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
            (-11, 1),
            (-4, -4),
            (-2, -2),
            (1, 11),
            (0, 11),
            (4, 4),
            (2, 2),
            (-11, 11),
        ]:
            if end < start or dend < dstart or (dend == 0 and dstart == 0) or dend == 0:
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
    @I.ir_module
    class RecipModule:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle):
            T.func_attr({"tir.noalias": True})
            n = T.int32(is_size_var=True)
            A = T.match_buffer(var_A, (n,))
            B = T.match_buffer(var_B, (n,))
            for i in range(n):
                with T.sblock("B"):
                    v_i = T.axis.spatial(n, i)
                    T.reads(A[v_i])
                    T.writes(B[v_i])
                    B[v_i] = T.float32(1.0) / (
                        T.float32(9999999999999999538762658202121142272.0) * A[v_i]
                    )

    f_recip = tvm.compile(RecipModule, target="llvm")

    for n in [4, 8, 16]:
        a = tvm.runtime.tensor(np.full((n,), 100, "float32"))
        b = tvm.runtime.empty((n,), "float32")
        f_recip(a, b)
        tvm.testing.assert_allclose(b.numpy(), np.zeros((n,), "float32"))

    @I.ir_module
    class SigmoidModule:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle):
            T.func_attr({"tir.noalias": True})
            n = T.int32(is_size_var=True)
            A = T.match_buffer(var_A, (n,))
            B = T.match_buffer(var_B, (n,))
            for i in range(n):
                with T.sblock("B"):
                    v_i = T.axis.spatial(n, i)
                    T.reads(A[v_i])
                    T.writes(B[v_i])
                    B[v_i] = T.sigmoid(A[v_i])

    f_sigmoid = tvm.compile(SigmoidModule, target="llvm")

    for n in [4, 8, 16]:
        a = tvm.runtime.tensor(np.full((n,), -1000, "float32"))
        b = tvm.runtime.empty((n,), "float32")
        f_sigmoid(a, b)
        tvm.testing.assert_allclose(b.numpy(), np.zeros((n,), "float32"))


@tvm.testing.requires_llvm
def test_dwarf_debug_information():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(
            A: T.Buffer((1024,), "float32"),
            B: T.Buffer((1024,), "float32"),
            C: T.Buffer((1024,), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            for i0_0 in T.parallel(256):
                for i0_1 in T.vectorized(4):
                    with T.sblock("C"):
                        v_i0 = T.axis.spatial(1024, i0_0 * 4 + i0_1)
                        T.reads(A[v_i0], B[v_i0])
                        T.writes(C[v_i0])
                        C[v_i0] = A[v_i0] + B[v_i0]

    def check_llvm_object():
        if tvm.target.codegen.llvm_version_major() < 5:
            return
        if tvm.target.codegen.llvm_version_major() > 6:
            return
        # build two functions
        mod = tvm.IRModule(
            {
                "fadd1": Module["main"].with_attr("global_symbol", "fadd1"),
                "fadd2": Module["main"].with_attr("global_symbol", "fadd2"),
            }
        )
        m = tvm.compile(mod, target="llvm")
        temp = utils.tempdir()
        o_path = temp.relpath("temp.o")
        m.write_to_file(o_path)
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
        mod = tvm.IRModule(
            {
                "fadd1": Module["main"].with_attr("global_symbol", "fadd1"),
                "fadd2": Module["main"].with_attr("global_symbol", "fadd2"),
            }
        )
        m = tvm.tir.build(mod, target={"kind": "llvm", "mtriple": "aarch64-linux-gnu"})
        ll = m.inspect_source("ll")

        # On non-Darwin OS, don't explicitly specify DWARF version.
        import re

        assert not re.search(r""""Dwarf Version""" "", ll)
        assert re.search(r"""llvm.dbg.value""", ll)

        # Try Darwin, require DWARF-2
        m = tvm.tir.build(mod, target={"kind": "llvm", "mtriple": "x86_64-apple-darwin-macho"})
        ll = m.inspect_source("ll")
        assert re.search(r"""i32 4, !"Dwarf Version", i32 2""", ll)
        assert re.search(r"""llvm.dbg.value""", ll)

    check_llvm_object()
    check_llvm_ir()


@tvm.testing.requires_llvm
def test_llvm_bf16():
    def dotest(do_vectorize):
        loop_kind = T.vectorized if do_vectorize else T.serial

        @I.ir_module
        class Module:
            @T.prim_func
            def main(
                A: T.Buffer((32,), "bfloat16"),
                B: T.Buffer((32,), "bfloat16"),
                D: T.Buffer((32,), "bfloat16"),
            ):
                T.func_attr({"tir.noalias": True})
                for x in loop_kind(32):
                    with T.sblock("D"):
                        v_x = T.axis.spatial(32, x)
                        T.reads(A[v_x], B[v_x])
                        T.writes(D[v_x])
                        D[v_x] = A[v_x] + B[v_x]

        np.random.seed(122)
        module = tvm.compile(Module, target="llvm")
        npa = np.random.rand(32).astype("bfloat16")
        npb = np.random.rand(32).astype("bfloat16")
        res = npa + npb
        a_ = tvm.runtime.tensor(npa)
        b_ = tvm.runtime.tensor(npb)
        c_ = tvm.runtime.empty((32,), "bfloat16")
        module(a_, b_, c_)
        # Note: directly compare without casting to float32 should work with the
        # latest numpy version.
        tvm.testing.assert_allclose(c_.numpy().astype("float32"), res.astype("float32"))

    dotest(True)
    dotest(False)


@tvm.testing.requires_llvm
def test_llvm_crt_static_lib():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(
            A: T.Buffer((32,), "bfloat16"),
            B: T.Buffer((32,), "bfloat16"),
            C: T.Buffer((32,), "bfloat16"),
        ):
            T.func_attr({"tir.noalias": True})
            for x in range(32):
                with T.sblock("compute"):
                    v_x = T.axis.spatial(32, x)
                    T.reads(A[v_x], B[v_x])
                    T.writes(C[v_x])
                    C[v_x] = A[v_x] + B[v_x]

    module = tvm.tir.build(
        Module.with_attr("system_lib_prefix", ""),
        target=tvm.target.Target("llvm"),
    )
    module.inspect_source()
    with utils.tempdir() as temp:
        module.write_to_file(temp.relpath("test.o"))


@tvm.testing.requires_llvm
def test_llvm_order_functions():
    """Check that functions in the LLVM module are ordered alphabetically."""

    # Note: the order is alphabetical because that's a predictable ordering. Any predictable
    # ordering will work fine, but if the ordering changes, this test will need to be updated.
    @I.ir_module
    class Module:
        @T.prim_func
        def Danny(v: T.float32) -> T.float32:
            T.ret(T.call_extern("float32", "Dave", v))

        @T.prim_func
        def Sammy(v: T.float32) -> T.float32:
            T.ret(T.call_extern("float32", "Eve", v))

        @T.prim_func
        def Kirby(v: T.float32) -> T.float32:
            T.ret(T.call_extern("float32", "Fred", v))

    ir_text = tvm.tir.build(Module, target="llvm").inspect_source("ll")
    # Skip functions whose names start with _.
    matches = re.findall(r"^define[^@]*@([a-zA-Z][a-zA-Z0-9_]*)", ir_text, re.MULTILINE)
    assert matches == sorted(matches)


@tvm.testing.requires_llvm
@tvm.testing.skip_if_32bit
def test_llvm_import():
    """all-platform-minimal-test: check shell dependent clang behavior."""
    # extern "C" is necessary to get the correct signature
    cc_code = """
    extern "C" float my_add(float x, float y) {
      return x + y;
    }
    """

    def check_llvm(use_file):
        if not clang.find_clang(required=False):
            print("skip because clang is not available")
            return
        temp = utils.tempdir()
        ll_path = temp.relpath("temp.ll")
        ll_code = clang.create_llvm(cc_code, output=ll_path)
        import_val = ll_path if use_file else ll_code

        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((10,), "float32"), B: T.Buffer((10,), "float32")):
                T.func_attr({"tir.noalias": True})
                for i in T.serial(10, annotations={"pragma_import_llvm": import_val}):
                    with T.sblock("B"):
                        v_i = T.axis.spatial(10, i)
                        T.reads(A[v_i])
                        T.writes(B[v_i])
                        B[v_i] = T.call_pure_extern("float32", "my_add", A[v_i], T.float32(1.0))

        f = tvm.compile(Module, target="llvm")
        dev = tvm.cpu(0)
        a = tvm.runtime.tensor(np.random.uniform(size=10).astype("float32"), dev)
        b = tvm.runtime.tensor(np.random.uniform(size=10).astype("float32"), dev)
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), a.numpy() + 1.0)

    check_llvm(use_file=True)
    check_llvm(use_file=False)


@tvm.testing.requires_llvm
def test_llvm_scalar_concat():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(x: T.int32, y: T.int32, buffer: T.Buffer((1,), "int32x2")):
            buffer[0] = T.Shuffle([x, y], [0, 1])

    # This will crash in LLVM codegen if CodeGenLLVM::CreateVecConcat doesn't convert
    # scalars to single-lane LLVM vectors.
    with tvm.transform.PassContext(config={"tir.disable_assert": True}):
        m = tvm.compile(Module, target="llvm")


@tvm.testing.requires_llvm
def test_raise_exception_during_codegen():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")) -> None:
            T.func_attr({"tir.noalias": True})
            for i in T.parallel(4):
                for j in T.parallel(4):
                    B[i, j] = A[i, j] * 2.0

    with pytest.raises(tvm.TVMError) as e:
        tvm.compile(Module, target="llvm")
    msg = str(e)
    assert msg.find("Nested parallel loop is not supported") != -1


@tvm.testing.requires_llvm
def test_llvm_target_attributes():
    """Check that when LLVM codegen creates new functions, they get the same target
    attributes as the original function.
    """

    @I.ir_module
    class Module:
        @T.prim_func
        def test_func(var_A: T.handle, var_B: T.handle, var_C: T.handle, tindex: T.int32):
            T.func_attr({"tir.noalias": True})
            A = T.match_buffer(var_A, (tindex,))
            B = T.match_buffer(var_B, (tindex,))
            C = T.match_buffer(var_C, (tindex,))
            for i in range(tindex):
                with T.sblock("B"):
                    v_i = T.axis.spatial(tindex, i)
                    T.reads(A[v_i])
                    T.writes(B[v_i])
                    B[v_i] = A[v_i]
            for i_0 in T.parallel(2):
                for i_1 in range((tindex + 1) // 2):
                    with T.sblock("C"):
                        v_i = T.axis.spatial(tindex, i_0 * ((tindex + 1) // 2) + i_1)
                        T.where(i_0 * ((tindex + 1) // 2) + i_1 < tindex)
                        T.reads(B[v_i])
                        T.writes(C[v_i])
                        C[v_i] = B[v_i] + T.float32(1.0)

    target_llvm = {
        "kind": "llvm",
        "mtriple": "x86_64-linux-gnu",
        "mcpu": "skylake",
        "mattr": ["+avx512f"],
    }
    target = tvm.target.Target(target_llvm, host=target_llvm)
    module = tvm.tir.build(Module, target=target)

    llvm_ir = module.inspect_source()
    llvm_ir_lines = llvm_ir.split("\n")

    attribute_definitions = dict()
    attributes_with_target = dict()
    functions_with_target = []

    for line in llvm_ir_lines:
        func_def = re.match(
            "define.* @(?P<func_name>[^(]*)[(].* #(?P<attr_num>[0-9]+) (!.* |){$", line
        )
        if func_def:
            functions_with_target.append(func_def.group("func_name"))
            attributes_with_target[func_def.group("attr_num")] = True
            continue
        attr_def = re.match("attributes #(?P<attr_num>[0-9]+) = {(?P<attr_list>.*)}", line)
        if attr_def:
            attribute_definitions[attr_def.group("attr_num")] = attr_def.group("attr_list")

    for k in list(attributes_with_target.keys()):
        assert re.match('.*"target-cpu"="skylake".*', attribute_definitions[k])
        assert re.match('.*"target-features"=".*[+]avx512f.*".*', attribute_definitions[k])

    expected_functions = [
        "__tvm_ffi_test_func",
        "__tvm_parallel_lambda",
    ]
    for n in expected_functions:
        assert n in functions_with_target


@tvm.testing.requires_llvm
def test_llvm_assume():
    """
    Check that LLVM does not error out when generating code with tir.assume.
    Verifying for llvm.assume being generated is not easy as the intrinsic and its
    related instructions get removed during optimizations
    """

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((4, 4), "int32"), B: T.Buffer((14,), "int32")):
            T.func_attr({"tir.noalias": True})
            A_1 = T.Buffer((16,), "int32", data=A.data)
            for axis0, axis1 in T.grid(4, 4):
                T.assume(axis0 < 3 or axis1 < 2 or A_1[axis0 * 4 + axis1] == 0)
            for i in range(14):
                B_1 = T.Buffer((14,), "int32", data=B.data)
                B_1[i] = A_1[i] * 2

    m = tvm.compile(Module, target="llvm")


@tvm.testing.requires_llvm
def test_debug_symbol_for_float64():
    """Check that LLVM can define DWARF debug type for float64

    In previous versions, only specific data types could exist in the
    function signature.  In this test, the "calling_conv" attribute
    prevents lowering to the PackedFunc API.
    """

    @I.ir_module
    class Module:
        @T.prim_func
        def main(a: T.handle("float64"), b: T.handle("float64"), n: T.int64):
            T.func_attr({"calling_conv": 2})
            A = T.Buffer(16, "float64", data=a)
            B = T.Buffer(16, "float64", data=b)
            for i in range(n):
                B[i] = A[i]

    tvm.compile(Module, target="llvm")


@tvm.testing.requires_llvm
def test_subroutine_call():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer(1, dtype="float32")):
            Module.subroutine(A.data)

        @T.prim_func
        def subroutine(A_data: T.handle("float32")):
            # The calling_conv parameter is to prevent MakePackedAPI
            # from changing the call signature of the subroutine.
            T.func_attr({"calling_conv": -1})
            A = T.decl_buffer(1, dtype="float32", data=A_data)
            A[0] = 42.0

    target = "llvm"
    dev = tvm.cpu()

    built = tvm.compile(Module)

    arr = tvm.runtime.tensor(np.zeros([1], "float32"), device=dev)
    built["main"](arr)
    assert arr.numpy()[0] == 42.0


@tvm.testing.requires_llvm
def test_call_packed_returning_void():
    """Allow codegen of PackedFunc calls returning void

    The LLVM codegen uses the CallNode's dtype to cast the return type
    of the PackedFunc into the appropriate LLVM output type.  However,
    there is no API type for `DataType::Void()`.  When the return type
    of a PackedFunc is void, the generated code should not attempt to
    read the return value.

    While `T.call_packed()` will produce a CallNode with an output
    dtype of "int32", the use of other return types is valid in TIR.
    This test case uses `T.Call` directly to allow an explicit dtype
    for the packed function call.
    """

    @I.ir_module
    class Module:
        @T.prim_func
        def main():
            T.Call(
                "void",
                tvm.ir.Op.get("tir.tvm_call_packed"),
                ["dummy_function_name"],
            )

    # Error occurred during build, as part of
    # CodeGenCPU::MakeCallPackedLowered.
    built = tvm.compile(Module, target="llvm")


@tvm.testing.requires_llvm
def test_call_packed_without_string_arg():
    """The first argument to tvm_call_packed must be a string

    Even if the invalid TIR is constructed, this should throw an
    exception to exit cleanly.  Previously, use of
    `args[0].as<StringImmNode>()` without a null check resulted in
    a segfault during codegen.
    """

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.Call("int32", tvm.ir.Op.get("tir.tvm_call_packed"), [A.data])

    with pytest.raises(tvm.TVMError):
        built = tvm.compile(Module, target="llvm")


@tvm.testing.requires_llvm
def test_call_extern_returning_void():
    """Like test_call_packed_returning_void, but for call_extern"""

    @I.ir_module
    class Module:
        @T.prim_func
        def main():
            T.Call("void", tvm.ir.Op.get("tir.call_extern"), ["dummy_function_name"])

    built = tvm.compile(Module, target="llvm")


def test_invalid_volatile_masked_buffer_load():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(b: T.handle):
            B = T.match_buffer(b, [4])
            a = T.allocate([4], "float32", scope="global")
            T.attr(a, "volatile_scope", 1)
            A = T.Buffer([4], data=a)
            B[0:4] = A.vload([T.Ramp(0, 1, 4)], predicate=T.Broadcast(T.bool(True), 4))

    err_msg = "The masked load intrinsic does not support declaring load as volatile."
    with pytest.raises(tvm.TVMError, match=err_msg):
        with tvm.target.Target("llvm"):
            tvm.compile(Module)


def test_invalid_volatile_masked_buffer_store():
    @I.ir_module
    class Module:
        @T.prim_func
        def main():
            a = T.allocate([4], "float32", scope="global")
            T.attr(a, "volatile_scope", 1)
            A = T.Buffer([4], data=a)
            A.vstore([T.Ramp(0, 1, 4)], T.Broadcast(0.0, 4), predicate=T.Broadcast(T.bool(True), 4))

    err_msg = "The masked store intrinsic does not support declaring store as volatile."
    with pytest.raises(tvm.TVMError, match=err_msg):
        with tvm.target.Target("llvm"):
            tvm.compile(Module)


def test_int_parameter():
    """Boolean may be passed to functions accepting int"""

    @I.ir_module
    class Module:
        @T.prim_func
        def main(arg: T.int32) -> T.int32:
            T.func_attr({"target": T.target("llvm")})
            if arg > 0:
                return 10
            else:
                return 20

    built = tvm.compile(Module)
    output = built(True)
    assert output == 10

    output = built(False)
    assert output == 20


def test_bool_parameter():
    """Integers may be passed to functions accepting bool"""

    @I.ir_module
    class Module:
        @T.prim_func
        def main(arg: T.bool) -> T.int32:
            T.func_attr({"target": T.target("llvm")})
            if arg:
                return 10
            else:
                return 20

    built = tvm.compile(Module)
    output = built(1)
    assert output == 10

    output = built(2)
    assert output == 10

    output = built(0)
    assert output == 20


def test_bool_return_value():
    """Booleans may be returned from a PrimFunc"""

    @I.ir_module
    class Module:
        @T.prim_func
        def main(value: T.int32) -> T.bool:
            T.func_attr({"target": T.target("llvm")})
            return value < 10

    built = tvm.compile(Module)
    assert isinstance(built(0), bool)
    assert built(0)

    assert isinstance(built(15), bool)
    assert not built(15)


def test_invalid_arguments():
    """Integers may be passed to functions accepting bool"""

    @I.ir_module
    class Module:
        @T.prim_func
        def main(a0: T.bool, a1: T.Buffer([10], "float32")) -> T.int32:
            T.func_attr({"target": T.target("llvm")})
            return 0

    built = tvm.compile(Module)
    with pytest.raises(RuntimeError):
        built(1, 1)

    with pytest.raises(RuntimeError):
        built(1, tvm.runtime.empty([10], "int32"))

    with pytest.raises(RuntimeError):
        built(False, tvm.runtime.empty([11], "float32"))


if __name__ == "__main__":
    tvm.testing.main()
