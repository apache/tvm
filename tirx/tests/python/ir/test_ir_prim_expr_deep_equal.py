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
import tvm_tirx


def test_equal_expr():
    x = tvm_tirx.ir.Var("x", tvm_tirx.ir.PrimType("int32"))
    y = tvm_tirx.ir.Var("y", tvm_tirx.ir.PrimType("int32"))

    def func1():
        return x + y + 1

    def func2():
        return tvm_tirx.ir.prim.truncdiv((x + y + 1) * y, 4)

    assert tvm_tirx.ir.prim.expr_deep_equal(func1(), func1())
    assert tvm_tirx.ir.prim.expr_deep_equal(func2(), func2())
    assert not tvm_tirx.ir.prim.expr_deep_equal(func2(), func1())


def test_structural_equal_maps_free_variables():
    x = tvm_tirx.ir.Var("x", tvm_tirx.ir.PrimType("int32"))
    y = tvm_tirx.ir.Var("y", tvm_tirx.ir.PrimType("int32"))
    tvm_tirx.ir.assert_structural_equal(x + 1, y + 1, map_free_vars=True)
    assert not tvm_tirx.ir.prim.expr_deep_equal(x + 1, y + 1)


if __name__ == "__main__":
    test_equal_expr()
