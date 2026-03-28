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


def test_verify_ssa():
    x = tvm.tirx.Var("x", "int32")
    y = tvm.tirx.Var("tindex", "int32")
    z = tvm.tirx.Evaluate(x + y)
    assert tvm.tirx.analysis.verify_ssa(tvm.tirx.PrimFunc([x, y], z))

    assert not tvm.tirx.analysis.verify_ssa(
        tvm.tirx.PrimFunc([x, y], tvm.tirx.SeqStmt([tvm.tirx.Bind(x, 1), z]))
    )


def test_verify_weak_let_ssa():
    x = tvm.tirx.Var("x", "int32")
    z1 = tvm.tirx.Let(x, 1, x + 1)
    z2 = tvm.tirx.Let(x, 2, x + 2)

    assert tvm.tirx.analysis.verify_ssa(tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(z1 + z1)))
    assert not tvm.tirx.analysis.verify_ssa(tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(z1 * z2)))


if __name__ == "__main__":
    test_verify_ssa()
    test_verify_weak_let_ssa()
