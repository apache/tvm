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
import tvm.testing
from tvm import te
from tvm.script import ir as I
from tvm.script import tir as T
import pytest


simple_target = tvm.target.Target("llvm -mtriple=x86_64-linux-gnu")
sve_target = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+sve")


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_loop(extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((16,), "float32")):
            T.func_attr({"target": target})
            for j in T.vectorized(0, extent):
                A[j] = 1

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((16,), "float32")):
            T.func_attr({"target": target})
            A[T.Ramp(0, 1, extent)] = T.Broadcast(1, extent)

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
            T.func_attr({"target": sve_target})
            for j in T.vectorized(T.vscale() * 4):
                A[j * 4 : j * 4 + 4] = T.Broadcast(T.float32(1), 4)

    error_msg = f"Creating scalable vectors from existing vectors is not supported."
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
        with tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+sve"):
            tvm.tir.transform.VectorizeLoop()(Module)


def test_vectorize_vector_scalable_error4():
    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(A: T.Buffer((25,), "float32")):
            T.func_attr({"target": sve_target})
            for j in T.vectorized(T.vscale() * 4):
                A[j * T.vscale() * 4 : j * T.vscale() * 4 + T.vscale() * 4] = T.Broadcast(
                    T.float32(1), T.vscale() * 4
                )

    error_msg = f"Creating scalable vectors from existing vectors is not supported."
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        with tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+sve"):
            tvm.tir.transform.VectorizeLoop()(Module)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_with_if(extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), n: T.int32, x: T.int32):
            T.func_attr({"target": target})
            for i in T.vectorized(extent):
                if x < n:
                    A[i] = A[i] + T.float32(1)
                else:
                    if i < n:
                        A[i] = T.float32(2)

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), n: T.int32, x: T.int32):
            T.func_attr({"target": target})
            if x < n:
                A[T.Ramp(0, 1, extent)] = A[T.Ramp(0, 1, extent)] + T.Broadcast(
                    T.float32(1), extent
                )
            else:
                for i_s in range(extent):
                    if i_s < n:
                        A[i_s] = T.float32(2)

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
            T.func_attr({"target": target})
            for i in T.vectorized(extent):
                v = A[i] + T.float32(1)
                A[i] = v + T.float32(2)

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32")):
            T.func_attr({"target": target})
            v = A[T.Ramp(0, 1, extent)] + T.Broadcast(T.float32(1), extent)
            A[T.Ramp(0, 1, extent)] = v + T.Broadcast(T.float32(2), extent)

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

    func = tvm.tir.PrimFunc([A, n], stmt).with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)
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

    func = tvm.tir.PrimFunc([A, n], stmt).with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    # Check that the loop wasn't vectorised
    assert isinstance(stmt, tvm.tir.For)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_if_then_else_scalarize(extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32")):
            T.func_attr({"target": target})
            for i in T.vectorized(extent):
                A[i] = T.if_then_else(i > 0, A[i] + T.float32(1), A[i])

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32")):
            T.func_attr({"target": target})
            for i_s in range(extent):
                A[i_s] = T.if_then_else(i_s > 0, A[i_s] + T.float32(1), A[i_s])

    mod = tvm.tir.transform.VectorizeLoop()(Before)
    tvm.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_if_then_else_vector(extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), n: T.int32):
            T.func_attr({"target": target})
            for i in range(n):
                for j in T.vectorized(extent):
                    A[i * extent + j] = T.if_then_else(i > 0, A[i * extent + j], 0)

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), n: T.int32):
            T.func_attr({"target": target})
            for i in range(n):
                A[T.Ramp(i * extent, 1, extent)] = T.if_then_else(
                    i > 0, A[T.Ramp(i * extent, 1, extent)], T.Broadcast(0, extent)
                )

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
            T.func_attr({"target": target})
            for i in T.vectorized(0, extent):
                B[i] = T.reinterpret("float32", A[i])

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((16,), "int32"), B: T.Buffer((16,), "float32")):
            T.func_attr({"target": target})
            B[T.Ramp(0, 1, extent)] = T.reinterpret(vec_str, A[T.Ramp(0, 1, extent)])

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
            T.func_attr({"target": target})
            for j in T.vectorized(extent):
                A[j] = op(T.float32(3), B[j])

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), B: T.Buffer((25,), "float32")):
            T.func_attr({"target": target})
            A[T.Ramp(0, 1, extent)] = op(T.Broadcast(T.float32(3), extent), B[T.Ramp(0, 1, extent)])

    mod = tvm.tir.transform.VectorizeLoop()(Before)
    tvm.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
@pytest.mark.parametrize("op", (T.And, T.Or))
def test_vectorize_logical(op, extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "bool"), B: T.Buffer((25,), "bool")):
            T.func_attr({"target": target})
            for j in T.vectorized(extent):
                A[j] = op(T.bool(1), B[j])

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "bool"), B: T.Buffer((25,), "bool")):
            T.func_attr({"target": target})
            A[T.Ramp(0, 1, extent)] = op(T.Broadcast(T.bool(1), extent), B[T.Ramp(0, 1, extent)])

    mod = tvm.tir.transform.VectorizeLoop()(Before)
    tvm.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_select(extent, target):
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), B: T.Buffer((25,), "float32")):
            T.func_attr({"target": target})
            for j in T.vectorized(extent):
                A[j] = T.Select(T.bool(True), A[j], B[j])

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "float32"), B: T.Buffer((25,), "float32")):
            T.func_attr({"target": target})
            A[T.Ramp(0, 1, extent)] = T.Select(
                T.Broadcast(T.bool(True), extent),
                A[T.Ramp(0, 1, extent)],
                B[T.Ramp(0, 1, extent)],
            )

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
            T.func_attr({"target": target})
            for j in T.vectorized(extent):
                A[j] = T.Cast("int32", B[j])

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((25,), "int32"), B: T.Buffer((25,), "float32")):
            T.func_attr({"target": target})
            A[T.Ramp(0, 1, extent)] = T.Cast(vec_str, B[T.Ramp(0, 1, extent)])

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
            T.func_attr({"target": simple_target})
            for j in T.vectorized(0, 4 * T.vscale()):
                A[j] = 13

    msg = (
        f"Failed to vectorize loop with extent T.vscale\\(\\) \\* 4 for target "
        f"llvm -keys=cpu -mtriple=x86_64-linux-gnu"
    )
    with pytest.raises(tvm.error.InternalError, match=msg):
        tvm.tir.transform.VectorizeLoop()(Mod)


if __name__ == "__main__":
    tvm.testing.main()
