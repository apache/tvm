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


def test_verify_ssa():
    x = te.var("x")
    y = te.var()
    z = tvm.tir.Evaluate(x + y)
    assert tvm.tir.analysis.verify_ssa(tvm.tir.PrimFunc([x, y], z))

    assert not tvm.tir.analysis.verify_ssa(tvm.tir.PrimFunc([x, y], tvm.tir.LetStmt(x, 1, z)))


def test_verify_weak_let_ssa():
    x = te.var("x")
    z1 = tvm.tir.Let(x, 1, x + 1)
    z2 = tvm.tir.Let(x, 2, x + 2)

    assert tvm.tir.analysis.verify_ssa(tvm.tir.PrimFunc([], tvm.tir.Evaluate(z1 + z1)))
    assert not tvm.tir.analysis.verify_ssa(tvm.tir.PrimFunc([], tvm.tir.Evaluate(z1 * z2)))


if __name__ == "__main__":
    test_verify_ssa()
    test_verify_weak_let_ssa()
