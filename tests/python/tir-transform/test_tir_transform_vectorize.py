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
from tvm import te


def test_vectorize_loop():
    dtype = "int64"
    n = te.var("n")
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
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


def test_vectorize_vector():
    dtype = "int64"
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


def test_vectorize_with_if():
    n = te.var("n")
    x = te.var("x")
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, 4, kind="vectorize") as i:
        with ib.if_scope(x < n):
            A[i] = A[i] + 1
        with ib.else_scope():
            with ib.if_scope(i < n):
                A[i] = 2.0
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n, x], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert isinstance(stmt, tvm.tir.IfThenElse)
    assert len(stmt.then_case.indices) == 1
    assert isinstance(stmt.then_case.indices[0], tvm.tir.Ramp)
    assert isinstance(stmt.then_case.value, tvm.tir.Add)
    assert stmt.then_case.value.dtype == "float32x4"
    assert isinstance(stmt.else_case, tvm.tir.For)


def test_vectorize_with_if_cond_int64():
    m = te.size_var("m", dtype="int64")
    A = te.placeholder((m,), name="A", dtype="float32")
    B = te.compute((m,), lambda i: te.if_then_else(i < 2, A[i], A[i] * 2), name="B")
    s = te.create_schedule(B.op)
    x, y = s[B].split(B.op.axis[0], factor=4)
    s[B].vectorize(y)
    f = tvm.build(s, [A, B], "llvm")


def test_vectorize_let():
    v = tvm.tir.Var("v", "float32")
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, 4, kind="vectorize") as i:
        ib.emit(lambda body: tvm.tir.LetStmt(v, A[i] + 1, body))
        A[i] = v + 2

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A], ib.get()))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body
    assert isinstance(stmt, tvm.tir.LetStmt)
    assert stmt.value.dtype == "float32x4"


def test_vectorize_with_le_cond():
    n = te.var("n")
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, 4, kind="vectorize") as i:
        with ib.if_scope(i <= n):
            A[i] = A[i] + 1
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert isinstance(stmt, tvm.tir.For)


def test_vectorize_with_ge_cond():
    n = te.var("n")
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, 4, kind="vectorize") as i:
        with ib.if_scope(i >= n):
            A[i] = A[i] + 1
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert isinstance(stmt, tvm.tir.For)


def test_vectorize_if_then_else():
    n = te.var("n")
    x = te.var("x")
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, 4, kind="vectorize") as i:
        A[i] = tvm.tir.call_intrin("float32", "tir.if_then_else", i > 0, A[i] + 1, A[i])
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n, x], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert isinstance(stmt, tvm.tir.For)

    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, n) as k:
        with ib.for_range(0, 4, kind="vectorize") as i:
            A[k * 4 + i] = tvm.tir.call_intrin(
                "float32", "tir.if_then_else", k > 0, A[k * 4 + i], 0
            )
    stmt = ib.get()

    assert isinstance(stmt.body, tvm.tir.For)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert not isinstance(stmt.body, tvm.tir.For)
    assert isinstance(stmt.body.value.args[2], tvm.tir.Broadcast)


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


if __name__ == "__main__":
    test_vectorize_vector()
    test_vectorize_with_if()
    test_vectorize_loop()
    test_vectorize_if_then_else()
    test_vectorize_with_le_cond()
    test_vectorize_with_ge_cond()
    test_vectorize_let()
    test_vectorize_while_fail()
    test_vectorize_dtype_mismatch()
