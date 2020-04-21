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
    x = te.var('x')
    y = te.var()
    z = tvm.tir.Evaluate(x + y)
    assert(tvm.tir.ir_pass.VerifySSA(z))


def test_convert_ssa():
    x = te.var('x')
    y = te.var()
    let1 = tvm.tir.Let(x, 1, x + 1)
    let2 = tvm.tir.Let(x, 1, x + y)
    z = tvm.tir.Evaluate(let1 + let2)
    assert(not tvm.tir.ir_pass.VerifySSA(z))
    z_ssa = tvm.tir.ir_pass.ConvertSSA(z)
    assert(tvm.tir.ir_pass.VerifySSA(z_ssa))


def test_expr_use_var():
    x = te.var('x')
    assert(tvm.tir.ir_pass.ExprUseVar(x+1, x))
    assert(not tvm.tir.ir_pass.ExprUseVar(1+10, x))


if __name__ == "__main__":
    test_expr_use_var()
