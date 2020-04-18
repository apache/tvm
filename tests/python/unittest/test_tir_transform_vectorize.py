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
    dtype = 'int64'
    n = te.var('n')
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, n) as i:
        with ib.for_range(0, 4, for_type="vectorize") as j:
            A[j] = tvm.tir.const(1, A.dtype)
    stmt = ib.get()

    assert isinstance(stmt.body, tvm.tir.For)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert isinstance(stmt, tvm.tir.For)
    assert not isinstance(stmt.body, tvm.tir.For)
    assert isinstance(stmt.body.index, tvm.tir.Ramp)
    assert isinstance(stmt.body.value, tvm.tir.Broadcast)


def test_vectorize_vector():
    dtype = 'int64'
    n = te.var('n')
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32x4", name="A")
    with ib.for_range(0, n) as i:
        with ib.for_range(0, 4, for_type="vectorize") as j:
            A[j] = tvm.tir.const(1, A.dtype)
    stmt = ib.get()
    assert isinstance(stmt.body, tvm.tir.For)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert isinstance(stmt, tvm.tir.For)
    assert not isinstance(stmt.body, tvm.tir.For)
    assert isinstance(stmt.body.index, tvm.tir.Ramp)
    assert isinstance(stmt.body.value, tvm.tir.Broadcast)


def test_vectorize_with_if():
    n = te.var('n')
    x = te.var('x')
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, 4, for_type="vectorize") as i:
        with ib.if_scope(x < n):
            A[i] = A[i] + 1
        with ib.else_scope():
            with ib.if_scope(i < n):
                A[i] = 2.0
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n, x], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert isinstance(stmt, tvm.tir.IfThenElse)
    assert isinstance(stmt.then_case.index, tvm.tir.Ramp)
    assert isinstance(stmt.then_case.value, tvm.tir.Add)
    assert stmt.then_case.value.dtype == "float32x4"
    assert isinstance(stmt.else_case, tvm.tir.For)


def test_vectorize_with_le_cond():
    n = te.var('n')
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, 4, for_type="vectorize") as i:
        with ib.if_scope(i <= n):
            A[i] = A[i] + 1
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert isinstance(stmt, tvm.tir.For)


def test_vectorize_with_ge_cond():
    n = te.var('n')
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, 4, for_type="vectorize") as i:
        with ib.if_scope(i >= n):
            A[i] = A[i] + 1
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert isinstance(stmt, tvm.tir.For)


def test_vectorize_if_then_else():
    n = te.var('n')
    x = te.var('x')
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, 4, for_type="vectorize") as i:
        A[i] = tvm.tir.call_intrin("float32", "tvm_if_then_else",
                               i > 0,
                               A[i] + 1, A[i])
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n, x], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert isinstance(stmt, tvm.tir.For)


    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, n) as k:
        with ib.for_range(0, 4, for_type="vectorize") as i:
            A[k * 4 + i] = tvm.tir.call_intrin("float32", "tvm_if_then_else",
                                           k > 0,
                                           A[k * 4 + i], 0)
    stmt = ib.get()

    assert isinstance(stmt.body, tvm.tir.For)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, n], stmt))
    stmt = tvm.tir.transform.VectorizeLoop()(mod)["main"].body

    assert not isinstance(stmt.body, tvm.tir.For)
    assert isinstance(stmt.body.value.args[2], tvm.tir.Broadcast)


if __name__ == "__main__":
    test_vectorize_vector()
    test_vectorize_with_if()
    test_vectorize_loop()
    test_vectorize_if_then_else()
    test_vectorize_with_le_cond()
    test_vectorize_with_ge_cond()
