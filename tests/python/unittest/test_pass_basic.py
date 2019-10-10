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

def test_simplify():
  tdiv = tvm.truncdiv
  tmod = tvm.truncmod
  x = tvm.var('x')
  e1 = tvm.ir_pass.Simplify(x + 2 + 1)
  assert(tvm.ir_pass.Equal(e1, x + 3))
  e2 = tvm.ir_pass.Simplify(x * 3 + 5 * x)
  assert(tvm.ir_pass.Equal(e2, x * 8))
  e3 = tvm.ir_pass.Simplify(x - tdiv(x, 3) * 3)
  assert(tvm.ir_pass.Equal(e3, tmod(x, 3)))


def test_verify_ssa():
    x = tvm.var('x')
    y = tvm.var()
    z = tvm.make.Evaluate(x + y)
    assert(tvm.ir_pass.VerifySSA(z))


def test_convert_ssa():
    x = tvm.var('x')
    y = tvm.var()
    let1 = tvm.make.Let(x, 1, x + 1)
    let2 = tvm.make.Let(x, 1, x + y)
    z = tvm.make.Evaluate(let1 + let2)
    assert(not tvm.ir_pass.VerifySSA(z))
    z_ssa = tvm.ir_pass.ConvertSSA(z)
    assert(tvm.ir_pass.VerifySSA(z_ssa))


def test_expr_use_var():
    x = tvm.var('x')
    assert(tvm.ir_pass.ExprUseVar(x+1, x))
    assert(not tvm.ir_pass.ExprUseVar(1+10, x))


if __name__ == "__main__":
    test_expr_use_var()
