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
import pytest

import tvm
import tvm.testing
from tvm import te
from tvm.script import ir as I
from tvm.script import tir as T

simple_target = tvm.target.Target("llvm -mtriple=x86_64-linux-gnu")
sve_target = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+sve")


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_loop(extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((16,), "float32")):
            for j in T.vectorized(0, extent):
                A[j] = 1

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((16,), "float32")):
            A[T.Ramp(0, 1, extent)] = T.Broadcast(1, extent)

    with tvm.target.Target(target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)


def test_vectorize_vector():
    n = te.var("n")
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32x4", name="A")
    with ib.for_range(0, n) as i:
        with ib.for_range(0, 4, kind="vectorize") as j:
            A[j] = tvm.tir.const(1, A.dtype)
    stmt = ib.get()
    assert isinstance(stmt.body, tvm.tir.For)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert isinstance(stmt, tvm.tir.For)
    assert not isinstance(stmt.body, tvm.tir.For)
    assert len(stmt.body.indices) == 1
    assert isinstance(stmt.body.indices[0], tvm.tir.Ramp)
    assert isinstance(stmt.body.value, tvm.tir.Broadcast)


def test_vectorize_vector_scalable_error():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32")):
            for j in T.vectorized(T.vscale() * 4):
                A[j * 4 : j * 4 + 4] = T.Broadcast(T.float32(1), 4)

    error_msg = f"Creating scalable vectors from existing vectors is not supported."
    with tvm.target.Target(sve_target):
        with pytest.raises(tvm.error.InternalError, match=error_msg):
            tvm.tir.transform.VectorizeLoop()(Module)


def test_vectorize_vector_scalable_error2():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32xvscalex4")):
            for j in T.vectorized(4):
                A[j] = T.Broadcast(T.float32(1), T.vscale() * 4)

    error_msg = f"Vectorizing over scalable buffer elements is not supported in vectorizer."
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        tvm.tir.transform.VectorizeLoop()(Module)


def test_vectorize_vector_scalable_error3():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32")):
            for j in T.vectorized(4):
                A[j * T.vscale() * 4 : j * T.vscale() * 4 + T.vscale() * 4] = T.Broadcast(
                    T.float32(1), T.vscale() * 4
                )

    error_msg = f"Vectorizing over existing scalable vectors is not supported."
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        with tvm.target.Target(sve_target):
            tvm.tir.transform.VectorizeLoop()(Module)


def test_vectorize_vector_scalable_error4():
    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(A: T.Buffer((25,), "float32")):
            for j in T.vectorized(T.vscale() * 4):
                A[j * T.vscale() * 4 : j * T.vscale() * 4 + T.vscale() * 4] = T.Broadcast(
                    T.float32(1), T.vscale() * 4
                )

    error_msg = f"Creating scalable vectors from existing vectors is not supported."
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        with tvm.target.Target(sve_target):
            tvm.tir.transform.VectorizeLoop()(Module)


def test_vectorize_with_if():
    extent = 4
    target = simple_target

    @I.ir_module
    class Before:
        @T.prim_func
        def main(a: T.handle, n: T.int32, x: T.int32):
            A = T.match_buffer(a, (25,), "float32")
            for i in T.vectorized(extent):
                if x < n:
                    A[i] = A[i] + T.float32(1)
                else:
                    if i < n:
                        A[i] = T.float32(2)

    @I.ir_module
    class After:
        @T.prim_func
        def main(a: T.handle, n: T.int32, x: T.int32):
            A = T.match_buffer(a, (25,), "float32")
            if x < n:
                A[T.Ramp(0, 1, extent)] = A[T.Ramp(0, 1, extent)] + T.Broadcast(
                    T.float32(1), extent
                )
            else:
                for i_s in range(extent):
                    if i_s < n:
                        A[i_s] = T.float32(2)

    with tvm.target.Target(target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)


def test_vectorize_if_scalable_extent():
    extent = T.vscale() * 4
    target = sve_target

    @I.ir_module
    class Before:
        @T.prim_func
        def main(a: T.handle, n: T.int32, x: T.int32):
            A = T.match_buffer(a, (25,), "float32")
            for i in T.vectorized(extent):
                if x < n:
                    A[i] = A[i] + T.float32(1)
                else:
                    if i < n:
                        A[i] = T.float32(2)

    @I.ir_module
    class After:
        @T.prim_func
        def main(a: T.handle, n: T.int32, x: T.int32):
            A = T.match_buffer(a, (25,), "float32")
            if x < n:
                A[T.Ramp(0, 1, extent)] = A[T.Ramp(0, 1, extent)] + T.Broadcast(
                    T.float32(1), extent
                )
            else:
                A.vstore(
                    [T.Ramp(0, 1, T.vscale() * 4)],
                    T.Broadcast(T.float32(2), T.vscale() * 4),
                    predicate=T.get_active_lane_mask("uint1xvscalex4", 0, n),
                )

    with tvm.target.Target(target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)


def test_vectorize_with_if_cond_int64():
    m = te.size_var("m", dtype="int64")
    A = te.placeholder((m,), name="A", dtype="float32")
    B = te.compute((m,), lambda i: te.if_then_else(i < 2, A[i], A[i] * 2), name="B")
    s = te.create_schedule(B.op)
    x, y = s[B].split(B.op.axis[0], factor=4)
    s[B].vectorize(y)
    f = tvm.build(s, [A, B], "llvm")


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_let(extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32")):
            for i in T.vectorized(extent):
                v = A[i] + T.float32(1)
                A[i] = v + T.float32(2)

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32")):
            v = A[T.Ramp(0, 1, extent)] + T.Broadcast(T.float32(1), extent)
            A[T.Ramp(0, 1, extent)] = v + T.Broadcast(T.float32(2), extent)

    with tvm.target.Target(target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (tvm.tir.vscale() * 4, sve_target)])
def test_vectorize_with_le_cond(extent, target):
    n = te.var("n")
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, extent, kind="vectorize") as i:
        with ib.if_scope(i <= n):
            A[i] = A[i] + 1
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n], stmt))

    with tvm.target.Target(target):
        stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

        # Check that the loop was't vectorised
        assert isinstance(stmt, tvm.tir.For)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (tvm.tir.vscale() * 4, sve_target)])
def test_vectorize_with_ge_cond(extent, target):
    n = te.var("n")
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, extent, kind="vectorize") as i:
        with ib.if_scope(i >= n):
            A[i] = A[i] + 1
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n], stmt))

    with tvm.target.Target(target):
        stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

        # Check that the loop wasn't vectorised
        assert isinstance(stmt, tvm.tir.For)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_if_then_else_scalarize(extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32")):
            for i in T.vectorized(extent):
                A[i] = T.if_then_else(i > 0, A[i] + T.float32(1), A[i])

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32")):
            for i_s in range(extent):
                A[i_s] = T.if_then_else(i_s > 0, A[i_s] + T.float32(1), A[i_s])

    with tvm.target.Target(target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_if_then_else_vector(extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), n: T.int32):
            for i in range(n):
                for j in T.vectorized(extent):
                    A[i * extent + j] = T.if_then_else(i > 0, A[i * extent + j], 0)

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), n: T.int32):
            for i in range(n):
                A[T.Ramp(i * extent, 1, extent)] = T.if_then_else(
                    i > 0, A[T.Ramp(i * extent, 1, extent)], T.Broadcast(0, extent)
                )

    with tvm.target.Target(target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)


def test_vectorize_let_if_then_else():
    @I.ir_module
    class Before:
        @T.prim_func
        def main():
            for i in T.vectorized(4):
                if i < 2:
                    result: T.int32 = T.if_then_else(i < 1, 1, 2)

    @I.ir_module
    class After:
        @T.prim_func
        def main():
            for i_s in range(4):
                if i_s < 2:
                    result: T.int32 = T.if_then_else(i_s < 1, 1, 2)
                    T.evaluate(0)

    with tvm.target.Target(simple_target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)


def test_vectorize_while_fail():
    """A while loop inside a vectorized loop should fail."""

    n = 64
    num_iter = 10

    def test_ir(A, B, C):
        ib = tvm.tir.ir_builder.create()
        n = C.shape[0]
        A = ib.buffer_ptr(A)
        B = ib.buffer_ptr(B)
        C = ib.buffer_ptr(C)
        i = ib.allocate("int32", (1,), name="i", scope="local")
        i[0] = 0

        with ib.for_range(0, n) as j:
            C[j] = 0.0

        with ib.for_range(0, n, kind="vectorize") as j:
            with ib.while_loop(i[0] < num_iter):
                C[j] += A[j] + B[j]
                i[0] += 1

        return ib.get()

    dtype = "float32"
    A = te.placeholder((n,), name="A", dtype=dtype)
    B = te.placeholder((n,), name="B", dtype=dtype)

    C = te.extern(
        (n,),
        [A, B],
        lambda ins, outs: test_ir(ins[0], ins[1], outs[0]),
        name="while_vectorize",
        dtype=dtype,
    )
    s = te.create_schedule(C.op)

    try:
        tvm.lower(s, [A, B, C], "llvm")
        assert False
    except tvm.error.TVMError as e:
        error_msg = str(e).split("\n")[-1]
        expected = "A while loop inside a vectorized loop not supported"
        assert expected in error_msg


def test_vectorize_dtype_mismatch():
    n = tvm.tir.IntImm("int64", 4)
    A = te.compute((n,), lambda i: tvm.tir.IntImm("int64", 2**31 - 1) + i, name="A")
    s = te.create_schedule(A.op)
    s[A].vectorize(A.op.axis[0])
    tvm.lower(s, [A], "llvm", simple_mode=True)


@pytest.mark.parametrize(
    "extent, vec_str, target",
    [(16, "float32x16", simple_target), (T.vscale() * 8, "float32xvscalex8", sve_target)],
)
def test_vectorize_with_reinterpret(extent, vec_str, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((16,), "int32"), B: T.Buffer((16,), "float32")):
            for i in T.vectorized(0, extent):
                B[i] = T.reinterpret("float32", A[i])

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((16,), "int32"), B: T.Buffer((16,), "float32")):
            B[T.Ramp(0, 1, extent)] = T.reinterpret(vec_str, A[T.Ramp(0, 1, extent)])

    with tvm.target.Target(target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
@pytest.mark.parametrize(
    "op",
    (
        T.Mul,
        T.Add,
        T.Sub,
        T.Div,
        T.Mod,
        T.FloorDiv,
        T.FloorMod,
        T.Min,
        T.Max,
        T.EQ,
        T.LT,
        T.LE,
        T.GE,
        T.GT,
        T.NE,
    ),
)
def test_vectorize_binary(op, extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), B: T.Buffer((25,), "float32")):
            for j in T.vectorized(extent):
                A[j] = op(T.float32(3), B[j])

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), B: T.Buffer((25,), "float32")):
            A[T.Ramp(0, 1, extent)] = op(T.Broadcast(T.float32(3), extent), B[T.Ramp(0, 1, extent)])

    with tvm.target.Target(target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
@pytest.mark.parametrize("op", (T.And, T.Or))
def test_vectorize_logical(op, extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "bool"), B: T.Buffer((25,), "bool")):
            for j in T.vectorized(extent):
                A[j] = op(T.bool(1), B[j])

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "bool"), B: T.Buffer((25,), "bool")):
            A[T.Ramp(0, 1, extent)] = op(T.Broadcast(T.bool(1), extent), B[T.Ramp(0, 1, extent)])

    with tvm.target.Target(target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_select(extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), B: T.Buffer((25,), "float32")):
            for j in T.vectorized(extent):
                A[j] = T.Select(T.bool(True), A[j], B[j])

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), B: T.Buffer((25,), "float32")):
            A[T.Ramp(0, 1, extent)] = T.Select(
                T.Broadcast(T.bool(True), extent),
                A[T.Ramp(0, 1, extent)],
                B[T.Ramp(0, 1, extent)],
            )

    with tvm.target.Target(target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize(
    "extent, vec_str, target",
    [(4, "int32x4", simple_target), (T.vscale() * 4, "int32xvscalex4", sve_target)],
)
def test_vectorize_cast(extent, vec_str, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "int32"), B: T.Buffer((25,), "float32")):
            for j in T.vectorized(extent):
                A[j] = T.Cast("int32", B[j])

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "int32"), B: T.Buffer((25,), "float32")):
            A[T.Ramp(0, 1, extent)] = T.Cast(vec_str, B[T.Ramp(0, 1, extent)])

    with tvm.target.Target(target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)


def test_illegal_extent():
    @I.ir_module(check_well_formed=False)
    class Mod:
        @T.prim_func
        def main(A: T.Buffer((25,), "int32")):
            n = T.Var("n", dtype="int32")
            for j in T.vectorized(n):
                A[j] = 3

    error_msg = f"Failed to vectorize loop with extent n for target \\(nullptr\\)"
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        tvm.tir.transform.VectorizeLoop()(Mod)


def test_illegal_vscale_in_non_sve_compilation():
    @I.ir_module
    class Mod:
        @T.prim_func
        def main(A: T.Buffer((16,), "float32")):
            for j in T.vectorized(0, 4 * T.vscale()):
                A[j] = 13

    msg = (
        f"Failed to vectorize loop with extent T.vscale\\(\\) \\* 4 for target "
        f"llvm -keys=cpu -mtriple=x86_64-linux-gnu"
    )
    with tvm.target.Target(simple_target):
        with pytest.raises(tvm.error.InternalError, match=msg):
            tvm.tir.transform.VectorizeLoop()(Mod)


def test_vectorize_and_predicate_all_buffer_loads_stores():
    @T.prim_func
    def before(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i_0 in T.serial(T.ceildiv(14, 4)):
            for i_1 in T.vectorized(4):
                if i_0 * 4 + i_1 < 14:
                    B[i_0 * 4 + i_1] = A[i_0 * 4 + i_1] + 1.0

    @T.prim_func
    def expected(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        for i_0 in range(4):
            load_a = T.meta_var(
                A.vload(
                    [T.Ramp(i_0 * 4, 1, 4)],
                    predicate=T.get_active_lane_mask("uint1x4", i_0 * 4, 14),
                )
            )
            add_1 = T.meta_var(load_a + T.Broadcast(T.float32(1), 4))
            B.vstore(
                [T.Ramp(i_0 * 4, 1, 4)],
                add_1,
                predicate=T.get_active_lane_mask("uint1x4", i_0 * 4, 14),
            )

    mod = tvm.IRModule.from_expr(before)
    with tvm.transform.PassContext(config={"tir.enable_buffer_level_predication": True}):
        after = tvm.tir.transform.VectorizeLoop()(mod)["main"]
    tvm.ir.assert_structural_equal(after, expected)


def test_vectorize_and_predicate_some_buffer_loads_stores():
    # Currently revert to scalarizing the block if not all accesses
    # have been predicated, otherwise incorrect code is generated.
    @T.prim_func
    def before(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i_0 in T.serial(T.ceildiv(14, 4)):
            for i_1 in T.vectorized(4):
                if i_0 * 4 + i_1 < 14:
                    B[i_0 * 4 + i_1] = A[i_0] + 1.0

    @T.prim_func
    def expected(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        for i_0, i_1_s in T.grid(4, 4):
            if i_0 * 4 + i_1_s < 14:
                B[i_0 * 4 + i_1_s] = A[i_0] + T.float32(1)

    mod = tvm.IRModule.from_expr(before)
    with tvm.transform.PassContext(config={"tir.enable_buffer_level_predication": True}):
        after = tvm.tir.transform.VectorizeLoop()(mod)["main"]
    tvm.ir.assert_structural_equal(after, expected)


def test_vectorize_and_predicate_multiple_access_statements():
    @T.prim_func
    def before(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i_0 in T.serial(T.ceildiv(14, 4)):
            for i_1 in T.vectorized(4):
                if i_0 * 4 + i_1 < 14:
                    A[i_0 * 4 + i_1] = 2.0
                    B[i_0 * 4 + i_1] = 1.0

    @T.prim_func
    def expected(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        for i_0 in range(4):
            A.vstore(
                [T.Ramp(i_0 * 4, 1, 4)],
                T.Broadcast(T.float32(2), 4),
                predicate=T.get_active_lane_mask("uint1x4", i_0 * 4, 14),
            )
            B.vstore(
                [T.Ramp(i_0 * 4, 1, 4)],
                T.Broadcast(T.float32(1), 4),
                predicate=T.get_active_lane_mask("uint1x4", i_0 * 4, 14),
            )

    before_mod = tvm.IRModule.from_expr(before)
    with tvm.transform.PassContext(config={"tir.enable_buffer_level_predication": True}):
        after = tvm.tir.transform.VectorizeLoop()(before_mod)["main"]
    tvm.ir.assert_structural_equal(after, expected)


def test_vectorize_and_predicate_invalid_conditions():
    @T.prim_func
    def before(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i_0 in T.serial(T.ceildiv(14, 4)):
            for i_1 in T.vectorized(4):
                if i_0 * 4 + i_1 > 14:
                    A[i_0 * 4 + i_1] = 2.0
                if 14 < i_0 * 4 + i_1:
                    A[i_0 * 4 + i_1] = 2.0
                if i_0 * 4 + i_1 < i_0 * 4 + i_1:
                    A[i_0 * 4 + i_1] = 2.0

    @T.prim_func
    def expected(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        for i_0 in range(4):
            for i_1_s in range(4):
                if i_0 * 4 + i_1_s > 14:
                    A[i_0 * 4 + i_1_s] = T.float32(2)
            for i_1_s in range(4):
                if 14 < i_0 * 4 + i_1_s:
                    A[i_0 * 4 + i_1_s] = T.float32(2)
            for i_1_s in range(4):
                if i_0 * 4 + i_1_s < i_0 * 4 + i_1_s:
                    A[i_0 * 4 + i_1_s] = T.float32(2)

    before_mod = tvm.IRModule.from_expr(before)
    with tvm.transform.PassContext(config={"tir.enable_buffer_level_predication": True}):
        after = tvm.tir.transform.VectorizeLoop()(before_mod)["main"]
    tvm.ir.assert_structural_equal(after, expected)


def test_vectorize_with_explicitly_disabled_buffer_level_predication():
    # Since the target has the SVE feature, buffer level predication is enabled
    # by default. However, it has been explicitly disabled by the pass context
    # option, so no buffer-level predicates should be added.
    @T.prim_func
    def before(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i_0 in T.serial(T.ceildiv(14, 4)):
            for i_1 in T.vectorized(4):
                if i_0 * 4 + i_1 < 14:
                    B[i_0 * 4 + i_1] = A[i_0 * 4 + i_1] + 1.0

    @T.prim_func
    def expected(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i_0, i_1_s in T.grid(4, 4):
            if i_0 * 4 + i_1_s < 14:
                B[i_0 * 4 + i_1_s] = A[i_0 * 4 + i_1_s] + T.float32(1)

    mod = tvm.IRModule.from_expr(before)
    with tvm.transform.PassContext(config={"tir.enable_buffer_level_predication": False}):
        with tvm.target.Target(sve_target):
            after = tvm.tir.transform.VectorizeLoop()(mod)["main"]
    tvm.ir.assert_structural_equal(after, expected)


def test_vectorize_and_predicate_buffer_load_stores_with_sve_func_attr_target():
    @T.prim_func
    def before(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "target": sve_target})
        for i_0 in T.serial(T.ceildiv(14, 4)):
            for i_1 in T.vectorized(4):
                if i_0 * 4 + i_1 < 14:
                    B[i_0 * 4 + i_1] = A[i_0 * 4 + i_1] + 1.0

    @T.prim_func
    def expected(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True), "target": sve_target})
        for i_0 in range(4):
            load_a = T.meta_var(
                A.vload(
                    [T.Ramp(i_0 * 4, 1, 4)],
                    predicate=T.get_active_lane_mask("uint1x4", i_0 * 4, 14),
                )
            )
            add_1 = T.meta_var(load_a + T.Broadcast(T.float32(1), 4))
            B.vstore(
                [T.Ramp(i_0 * 4, 1, 4)],
                add_1,
                predicate=T.get_active_lane_mask("uint1x4", i_0 * 4, 14),
            )

    mod = tvm.IRModule.from_expr(before)
    after = tvm.tir.transform.VectorizeLoop()(mod)["main"]
    tvm.ir.assert_structural_equal(after, expected)


def test_vectorize_and_predicate_buffer_load_stores_with_sve_attr_scope_target():
    @T.prim_func
    def before(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        with T.attr(sve_target, "target", 0):
            for i_0 in T.serial(T.ceildiv(14, 4)):
                for i_1 in T.vectorized(4):
                    if i_0 * 4 + i_1 < 14:
                        B[i_0 * 4 + i_1] = A[i_0 * 4 + i_1] + 1.0

    @T.prim_func
    def expected(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        with T.attr(sve_target, "target", 0):
            for i_0 in range(4):
                load_a = T.meta_var(
                    A.vload(
                        [T.Ramp(i_0 * 4, 1, 4)],
                        predicate=T.get_active_lane_mask("uint1x4", i_0 * 4, 14),
                    )
                )
                add_1 = T.meta_var(load_a + T.Broadcast(T.float32(1), 4))
                B.vstore(
                    [T.Ramp(i_0 * 4, 1, 4)],
                    add_1,
                    predicate=T.get_active_lane_mask("uint1x4", i_0 * 4, 14),
                )

    mod = tvm.IRModule.from_expr(before)
    after = tvm.tir.transform.VectorizeLoop()(mod)["main"]
    tvm.ir.assert_structural_equal(after, expected)


@pytest.mark.parametrize(
    "extent, vec_str, target",
    [(4, "float32x4", simple_target)],
)
def test_vectorize_llvm_pure_intrin(extent, vec_str, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), B: T.Buffer((25,), "float32")):
            for j in T.vectorized(extent):
                A[j] = T.call_llvm_pure_intrin(
                    "float32", "llvm.sqrt", tvm.tir.const(1, "uint"), B[j]
                )

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), B: T.Buffer((25,), "float32")):
            A[T.Ramp(0, 1, extent)] = T.call_llvm_pure_intrin(
                vec_str, "llvm.sqrt", tvm.tir.const(1, "uint"), B[T.Ramp(0, 1, extent)]
            )

    with tvm.target.Target(target):
        mod = tvm.tir.transform.VectorizeLoop()(Before)
        tvm.ir.assert_structural_equal(mod, After)
        mod = tvm.build(mod, target)


@pytest.mark.parametrize(
    "extent, vec_str, target",
    [(4, "int32x4", simple_target)],
)
def test_vectorize_llvm_pure_intrin_fail(extent, vec_str, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "int32"), B: T.Buffer((25,), "float32")):
            for j in T.vectorized(extent):
                A[j] = T.call_llvm_pure_intrin(
                    "int32", "llvm.lround", tvm.tir.const(1, "uint"), B[j]
                )

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "int32"), B: T.Buffer((25,), "float32")):
            A[T.Ramp(0, 1, extent)] = T.call_llvm_pure_intrin(
                vec_str, "llvm.lround", tvm.tir.const(1, "uint"), B[T.Ramp(0, 1, extent)]
            )

    with pytest.raises(Exception) as e_info:
        with tvm.target.Target(target):
            mod = tvm.tir.transform.VectorizeLoop()(Before)
            ex = tvm.build(mod, target)
    tvm.ir.assert_structural_equal(mod, After)
    assert "Intrinsic does not support vectors" in e_info.value.args[0]


if __name__ == "__main__":
    tvm.testing.main()
