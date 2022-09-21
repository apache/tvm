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
from tvm.script import tir as T
import tvm.testing


def nop():
    return tvm.tir.Evaluate(0)


def test_remove_no_op():
    i = te.var("i")
    j = te.var("j")
    k = te.var("k")
    m = te.var("m")
    n = te.var("n")
    dtype = "int64"
    Ab = tvm.tir.decl_buffer((n,), dtype)
    stmt = tvm.tir.For(
        i,
        0,
        4,
        tvm.tir.ForKind.SERIAL,
        tvm.tir.For(
            j,
            0,
            n,
            tvm.tir.ForKind.SERIAL,
            tvm.tir.For(
                k,
                0,
                m,
                tvm.tir.ForKind.SERIAL,
                tvm.tir.IfThenElse((i * m + j + k < n), tvm.tir.Evaluate(m), tvm.tir.Evaluate(n)),
            ),
        ),
    )

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt))
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body

    assert isinstance(ret, tvm.tir.Evaluate)
    store = tvm.tir.BufferStore(Ab, tvm.tir.BufferLoad(Ab, [i]) + 1, [i + 1])
    stmt2 = tvm.tir.SeqStmt([nop(), tvm.tir.SeqStmt([store, nop()])])

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt2))
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body
    assert ret == store

    # remove zero extent loop
    stmt3 = tvm.tir.For(i, 0, 0, tvm.tir.ForKind.SERIAL, store)
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt3))
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body
    assert isinstance(ret, tvm.tir.Evaluate)


def test_remove_no_op_with_invalid_extent():
    @T.prim_func
    def main(A: T.Buffer[(16), "int32"], B: T.Buffer[(16), "int32"]) -> None:
        for i in T.serial(16):
            for j in T.serial(i - 20):
                B[i] = A[i] + j

    mod = tvm.ir.module.IRModule.from_expr(main)
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body
    assert isinstance(ret, tvm.tir.Evaluate)


if __name__ == "__main__":
    tvm.testing.main()
