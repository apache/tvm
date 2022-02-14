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

# A test program which gives the opportunity for the CSE pass to introduce two new variables, at two different levels
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
    body = tvm.tir.LetStmt(
        z1,
        1,
        tvm.tir.LetStmt(
            z2,
            2,
            tvm.tir.SeqStmt(
                [
                    tvm.tir.Store(buffer.data, z1 + z2, i1),
                    tvm.tir.LetStmt(
                        x,
                        1,
                        tvm.tir.LetStmt(
                            y,
                            1,
                            tvm.tir.LetStmt(
                                a,
                                (x + y) + (z1 + z2),
                                tvm.tir.LetStmt(
                                    b, (x + y) + z3, tvm.tir.Store(buffer.data, a + b, i2)
                                ),
                            ),
                        ),
                    ),
                ]
            ),
        ),
    )
    # This test program gives the opportunity to introduce two new variables, at two different levels
    # and to perform replacements in the value of "a" and "b", using these new variables
    # We will check all of that underneath and more, making also sure that nothing else has been changed

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([i1, i2, z3], body))
    body = tvm.tir.transform.CommonSubexprElimTIR()(mod)

    tvm.transform.PrintIR()(body)

    body = body["main"].body  # Gets the body of the main, i.e. the full statement

    assert body.var.name == "z1"
    assert body.value == 1

    body = body.body

    assert body.var.name == "z2"
    assert body.value == 2
    # This is the let-in for the first variable generated cse_var_1
    assert isinstance(body.body, tvm.tir.LetStmt)

    body = body.body

    # And this is the name and value of this variable
    cse_var_1 = body.var  # Keep the variable accessible for later checking the replacements
    assert body.var.name == "cse_var_1"
    assert tvm.ir.structural_equal(body.value, z1 + z2)
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
    cse_var_2 = body.var  # Keep the variable accessible for later checking the replacements
    assert body.var.name == "cse_var_2"
    assert tvm.ir.structural_equal(body.value, x + y)

    body = body.body

    body.var.name == "a"
    # Check that the replacement has been done correctly!
    assert tvm.ir.structural_equal(body.value, cse_var_2 + cse_var_1)

    body = body.body

    body.var.name == "b"
    # Check that the replacement has been done correctly!
    assert tvm.ir.structural_equal(body.value, cse_var_2 + z3)

    assert isinstance(body.body, tvm.tir.Store)


# First specific test for if nodes : Some duplicated computations appear only in one branch (here the Then branch), not in both branches.
# In this case, the CSE pass should introduce the redundant computation at the top if the Then branch, not before the whole If
# (otherwise that would lead to some computations being computed for nothing when it is the Else branch that is executed).
def test_cse_ifNode_1():
    b = te.var("b")
    i1 = te.var("i1")
    i2 = te.var("i2")
    i3 = te.var("i3")
    y = te.var("y")
    z = te.var("z")
    dtype = "int32"
    buffer = tvm.tir.decl_buffer((50,), dtype)
    # Test prog :
    # let b=1 in
    #   if(b) {
    #      Mem[i1] = y+z
    #     Mem[i2] = y+z
    #   }
    #   else {
    #     Mem[i3] = y
    #   }
    body = tvm.tir.LetStmt(
        b,
        1,
        tvm.tir.IfThenElse(
            b,
            tvm.tir.SeqStmt(
                [tvm.tir.Store(buffer.data, y + z, i1), tvm.tir.Store(buffer.data, y + z, i2)]
            ),
            tvm.tir.Store(buffer.data, y, i3),
        ),
    )

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([i1, i2, i3, y, z], body))
    body = tvm.tir.transform.CommonSubexprElimTIR()(mod)

    tvm.transform.PrintIR()(body)

    body = body["main"].body  # Gets the body of the main, i.e. the full statement

    assert body.var.name == "b"
    assert body.value == 1
    assert isinstance(body.body, tvm.tir.IfThenElse)

    body = body.body

    assert isinstance(body.then_case, tvm.tir.LetStmt)

    body = body.then_case

    # The let-in introduced by the CSE should appear now, inside the Then branch of the If node
    assert body.var.name == "cse_var_1"
    # and it should contain the expression (y+z) that was redundant
    assert tvm.ir.structural_equal(body.value, y + z)


# Second test for if nodes : Some duplicated computations appear in both the Then and the Else branch.
# In this case, the CSE pass should introduce the redundant computation before the whole If node, because
# regardless of the execution path, it is going to be computed.
def test_cse_ifNode_2():
    b = te.var("b")
    i1 = te.var("i1")
    i2 = te.var("i2")
    i3 = te.var("i3")
    y = te.var("y")
    z = te.var("z")
    dtype = "int32"
    buffer = tvm.tir.decl_buffer((50,), dtype)
    # Test prog :
    # let b=1 in
    #   if(b) {
    #     Mem[i1] = y+z
    #      Mem[i2] = y
    #   }
    #   else {
    #     Mem[i3] = y+z
    #   }
    body = tvm.tir.LetStmt(
        b,
        1,
        tvm.tir.IfThenElse(
            b,
            tvm.tir.SeqStmt(
                [
                    tvm.tir.Store(buffer.data, y + z, i1),  # (y+z) is present in the Then branch
                    tvm.tir.Store(buffer.data, y, i2),
                ]
            ),
            tvm.tir.Store(buffer.data, y + z, i3),  # and also present in the Else branch
        ),
    )

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([i1, i2, i3, y, z], body))
    body = tvm.tir.transform.CommonSubexprElimTIR()(mod)

    tvm.transform.PrintIR()(body)

    body = body["main"].body  # Gets the body of the main, i.e. the full statement

    assert isinstance(body, tvm.tir.LetStmt)

    # The let-in introduced by the CSE should appear now, at the toplevel (i.e. before the If)
    assert body.var.name == "cse_var_1"
    # and it should contain the expression (y+z) that was redundant
    assert tvm.ir.structural_equal(body.value, y + z)


# Test commoning in cascade : after having introduced a big exp ((x+y)+z) into a new variable,
# it will become possible to do another commoning for (x+y) which appears both in the new variable
# and in the rest of the program.
def test_cse_cascade():
    i1 = te.var("i1")
    i2 = te.var("i2")
    i3 = te.var("i3")
    x = te.var("x")
    y = te.var("y")
    z = te.var("z")
    dtype = "int32"
    buffer = tvm.tir.decl_buffer((50,), dtype)
    # Test prog :
    # Mem[i1] = (x+y)+z;
    # Mem[i2] = (x+y)+z;
    # Mem[i3] = x+y
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.Store(buffer.data, (x + y) + z, i1),
            tvm.tir.Store(buffer.data, (x + y) + z, i2),
            tvm.tir.Store(buffer.data, (x + y), i3),
        ]
    )

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([i1, i2, i3, x, y, z], body))
    body = tvm.tir.transform.CommonSubexprElimTIR()(mod)

    tvm.transform.PrintIR()(body)

    body = body["main"].body  # Gets the body of the main, i.e. the full statement

    assert isinstance(body, tvm.tir.LetStmt)

    # The second let-in (by order introduced) introduced by the CSE should appear first
    cse_var_2 = body.var  # Keep the variable accessible for later checking the replacements
    assert body.var.name == "cse_var_2"
    # and it should contain the expression (x+y)
    assert tvm.ir.structural_equal(body.value, (x + y))

    body = body.body

    assert isinstance(body, tvm.tir.LetStmt)

    # The first let-in (by order introduced) introduced by the CSE should appear now, after the 2nd
    cse_var_1 = body.var  # Keep the variable accessible for later checking the replacements
    assert body.var.name == "cse_var_1"
    # and it should contain the expression cse_var_2+z
    assert tvm.ir.structural_equal(body.value, cse_var_2 + z)

    body = body.body

    assert isinstance(body, tvm.tir.SeqStmt)
    assert isinstance(body[0], tvm.tir.Store)
    assert isinstance(body[1], tvm.tir.Store)
    assert isinstance(body[2], tvm.tir.Store)

    store1 = body[0]
    store2 = body[1]
    store3 = body[2]

    assert tvm.ir.structural_equal(store1.value, cse_var_1)
    assert tvm.ir.structural_equal(store2.value, cse_var_1)
    assert tvm.ir.structural_equal(store3.value, cse_var_2)


if __name__ == "__main__":
    test_cse()
    test_cse_ifNode_1()
    test_cse_ifNode_2()
    test_cse_cascade()
