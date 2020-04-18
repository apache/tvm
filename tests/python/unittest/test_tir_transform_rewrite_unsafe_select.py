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


def test_rewrite_Select():
    ib = tvm.tir.ir_builder.create()
    A = ib.allocate("float32", 100, name="A", scope="global")
    i = te.var("i")
    y = tvm.tir.Select(i > 1, A[i-1], 1.0)

    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([i], tvm.tir.Evaluate(y)))
    yy = tvm.tir.transform.RewriteUnsafeSelect()(mod)["main"].body.value

    z = tvm.tir.Select(
        tvm.tir.Select(i > 1, A[i-1], 1.0) > 0.0, A[i], 0.1)
    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([i], tvm.tir.Evaluate(z)))
    zz = tvm.tir.transform.RewriteUnsafeSelect()(mod)["main"].body.value

    a = tvm.tir.Select(tvm.tir.floordiv(i, 4) > 10, y, z)

    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([i], tvm.tir.Evaluate(a)))
    aa = tvm.tir.transform.RewriteUnsafeSelect()(mod)["main"].body.value
    assert yy.name == "tvm_if_then_else"
    assert zz.name == "tvm_if_then_else"
    assert isinstance(aa, tvm.tir.Select)


if __name__ == "__main__":
    test_rewrite_Select()
