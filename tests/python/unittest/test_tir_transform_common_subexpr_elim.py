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


def test_cse():
    z1 = te.var("z1")
    z2 = te.var("z2")
    z3 = te.var("z3")
    i1 = te.var("i1")
    i2 = te.var("i2")
    x = te.var("x")
    y = te.var("y")
    a = te.var("a")
    b = te.var("b")
    dtype = "int32"
    buffer = tvm.tir.decl_buffer((50,), dtype)
    # Test prog : 
    # let z1=1 in let z2=2 in
    #   Mem[i1] = z1+z2;
    #   let x = 1 in let y = 1 in 
    #     let a = (x+y) + (z1+z2) in
    #       let b = (x+y) + z3 in
    #         Mem[i2] = a+b;
    body = tvm.tir.LetStmt(z1, 1, 
                            tvm.tir.LetStmt(z2, 2, 
                                            tvm.tir.SeqStmt([tvm.tir.Store(buffer.data, z1+z2, i1),
                                                            tvm.tir.LetStmt(x, 1, 
                                                                            tvm.tir.LetStmt(y, 1, 
                                                                                            tvm.tir.LetStmt(a, (x+y) + (z1+z2), 
                                                                                                            tvm.tir.LetStmt(b, (x+y) + z3, 
                                                                                                            tvm.tir.Store(buffer.data, a+b, i2)
                                                                                                          )
                                                                                            )
                                                                              )
                                                                            )
                                                          ])
                                            )
                            )
    # This test program gives the opportunity to introduce two new variables, at two different levels
    # and to perform replacements in the value of "a" and "b", using these new variables
    # We will check all of that underneath and more, making also sure that nothing else has been changed

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([i1, i2, z3], body))
    body = tvm.tir.transform.CommonSubexprElim()(mod)

    tvm.transform.PrintIR()(body)

    body = body["main"].body # Gets the body of the main, i.e. the full statement

    assert body.var.name == "z1"
    assert body.value == 1

    body = body.body

    assert body.var.name == "z2"
    assert body.value == 2

    # This is the let-in for the first variable generated cse_var_1
    assert isinstance(body.body, tvm.tir.LetStmt)

    body = body.body

    # And this is the name and value of this variable
    cse_var_1 = body.var # Keep the variable accessible for later checking the replacements
    assert body.var.name == "cse_var_1"
    assert tvm.ir.structural_equal(body.value, z1+z2)

    assert isinstance(body.body, tvm.tir.SeqStmt)

    body = body.body

    assert isinstance(body[0], tvm.tir.Store)
    assert isinstance(body[1], tvm.tir.LetStmt)

    body = body[1]

    assert body.var.name == "x"
    assert body.value == 1

    body = body.body

    assert body.var.name == "y"
    assert body.value == 1

    # This is the let-in for the second variable generated cse_var_2
    assert isinstance(body.body, tvm.tir.LetStmt)

    body = body.body

    # And this is the name and value of this variable
    cse_var_2 = body.var # Keep the variable accessible for later checking the replacements
    assert body.var.name == "cse_var_2"
    assert tvm.ir.structural_equal(body.value, x+y)

    body = body.body

    body.var.name == "a"
    # Check that the replacement has been done correctly!
    assert tvm.ir.structural_equal(body.value, cse_var_2+cse_var_1)
    
    body = body.body

    body.var.name == "b"
    # Check that the replacement has been done correctly!
    assert tvm.ir.structural_equal(body.value, cse_var_2+z3)

    assert isinstance(body.body, tvm.tir.Store)


if __name__ == "__main__":
    test_cse()
