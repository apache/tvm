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
from tvm.script import tir as T


def _count_alloc(stmt):
    num_alloc = [0]

    def visit(n):
        if isinstance(n, tvm.tir.Allocate):
            num_alloc[0] += 1

    tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    return num_alloc[0]


def _count_ptx_ldg32(stmt):
    num_call = [0]

    def visit(n):
        if isinstance(n, tvm.tir.Call) and n.op.name == "tir.ptx_ldg32":
            num_call[0] += 1

    tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    return num_call[0]


@T.prim_func
def where_no_alloc(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "target": T.target("cuda")})
    for i in range(4):
        C[i] = T.if_then_else(A[i] > T.float32(0), A[i], T.float32(0))


@T.prim_func
def where_no_alloc_cpu(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "target": T.target("llvm")})
    for i in range(4):
        C[i] = T.if_then_else(A[i] > T.float32(0), A[i], T.float32(0))


def test_inject_ptx_ldg32_inserts_alloc_for_no_alloc_func():
    mod = tvm.IRModule.from_expr(where_no_alloc)
    assert _count_alloc(mod["main"].body) == 0

    mod = tvm.tir.transform.InjectPTXLDG32()(mod)
    assert _count_alloc(mod["main"].body) > 0
    assert _count_ptx_ldg32(mod["main"].body) == 1


def test_inject_ptx_ldg32_skip_non_cuda_target():
    mod = tvm.IRModule.from_expr(where_no_alloc_cpu)
    cpu_target = tvm.target.Target("llvm")
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", cpu_target)})
    assert _count_alloc(mod["main"].body) == 0

    mod = tvm.tir.transform.InjectPTXLDG32()(mod)
    assert _count_alloc(mod["main"].body) == 0
    assert _count_ptx_ldg32(mod["main"].body) == 0


if __name__ == "__main__":
    tvm.testing.main()
