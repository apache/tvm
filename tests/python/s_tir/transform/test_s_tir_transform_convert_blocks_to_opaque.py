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
# ruff: noqa: F401
import pytest

import tvm
import tvm.testing
from tvm import s_tir, te, tirx
from tvm.script import ir as I
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tvm.s_tir.transform.StmtSimplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed.with_attr("global_symbol", "main"))


@Ts.prim_func
def elementwise_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with Ts.sblock():
            Ts.reads(A[i, 0:16])
            Ts.writes(C[i, 0:16])
            B = Ts.sblock_alloc_buffer((16, 16), "float32")
            for j in range(0, 16):
                with Ts.sblock():
                    vi = Ts.axis.S(16, i)
                    vj = Ts.axis.S(16, j)
                    B[vi, vj] = A[vi, vj] + 1.0
            for j in range(0, 16):
                with Ts.sblock():
                    vi = Ts.axis.S(16, i)
                    vj = Ts.axis.S(16, j)
                    C[vi, vj] = B[vi, vj] * 2.0


@Ts.prim_func
def substituted_elementwise_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with Ts.sblock():
            Ts.reads(A[i, 0:16])
            Ts.writes(C[i, 0:16])
            B = Ts.sblock_alloc_buffer([16, 16], "float32")
            for j in range(0, 16):
                with Ts.sblock():
                    Ts.reads([A[i, j]])
                    Ts.writes([B[i, j]])
                    B[i, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with Ts.sblock():
                    Ts.reads([B[i, j]])
                    Ts.writes([C[i, j]])
                    C[i, j] = B[i, j] * 2.0


def test_elementwise():
    _check(elementwise_func, substituted_elementwise_func)


def test_error_if_predicate_uses_block_variables():
    @I.ir_module(check_well_formed=False)
    class Before:
        @Ts.prim_func
        def main(A: T.Buffer(8, "int32")):
            for i in T.serial(8):
                with Ts.sblock():
                    vi = Ts.axis.remap("S", [i])
                    Ts.where(vi < 6)
                    T.evaluate(0)

    with pytest.raises(RuntimeError):
        tvm.s_tir.transform.ConvertBlocksToOpaque()(Before)


if __name__ == "__main__":
    tvm.testing.main()
