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
import hashlib

import tvm
from tvm import auto_scheduler, te, topi
from tvm.ir.base import save_json
from tvm.ir.module import IRModule
from tvm.script import tir as T


# -----------------------------------------------------
# Basic test for the expected Behavior of the CSE pass
# -----------------------------------------------------
# A test program which gives the opportunity for the CSE pass to introduce two new variables,
# at two different levels
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
                    tvm.tir.BufferStore(buffer, z1 + z2, [i1]),
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
                                    b, (x + y) + z3, tvm.tir.BufferStore(buffer, a + b, [i2])
                                ),
                            ),
                        ),
                    ),
                ]
            ),
        ),
    )
    # This test program gives the opportunity to introduce two new variables, at two different
    # levels and to perform replacements in the value of "a" and "b", using these new variables.
    # We will check all of that underneath and more, making also sure that nothing else has changed

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

    assert isinstance(body[0], tvm.tir.BufferStore)
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

    assert isinstance(body.body, tvm.tir.BufferStore)


# -----------------------------------------------------
# Tests related to If nodes
# -----------------------------------------------------
# First specific test for if nodes : Some duplicated computations appear only in one branch (here
# the Then branch), not in both branches.
# In this case, the CSE pass should introduce the redundant computation at the top of the Then
# branch, not before the whole If (otherwise that would lead to some computations being computed
# for nothing when it is the Else branch that is executed).
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
                [tvm.tir.BufferStore(buffer, y + z, [i1]), tvm.tir.BufferStore(buffer, y + z, [i2])]
            ),
            tvm.tir.BufferStore(buffer, y, [i3]),
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


# Second test for if nodes : Some duplicated computations appear in both the Then and Else branch.
# In this case, the CSE pass should introduce the redundant computation before the whole If node,
# because regardless of the execution path, it is going to be computed.
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
                    tvm.tir.BufferStore(buffer, y + z, [i1]),  # (y+z) is present in Then branch
                    tvm.tir.BufferStore(buffer, y, [i2]),
                ]
            ),
            tvm.tir.BufferStore(buffer, y + z, [i3]),  # and also present in the Else branch
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


# -------------------------------------------------------------------------------------------------
# Test commoning in cascade : after having introduced a big exp ((x+y)+z) into a new variable,
# it will become possible to do another commoning for (x+y) which appears both in the new variable
# and in the rest of the program.
# -------------------------------------------------------------------------------------------------
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
            tvm.tir.BufferStore(buffer, (x + y) + z, [i1]),
            tvm.tir.BufferStore(buffer, (x + y) + z, [i2]),
            tvm.tir.BufferStore(buffer, (x + y), [i3]),
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
    assert isinstance(body[0], tvm.tir.BufferStore)
    assert isinstance(body[1], tvm.tir.BufferStore)
    assert isinstance(body[2], tvm.tir.BufferStore)

    store1 = body[0]
    store2 = body[1]
    store3 = body[2]

    assert tvm.ir.structural_equal(store1.value, cse_var_1)
    assert tvm.ir.structural_equal(store2.value, cse_var_1)
    assert tvm.ir.structural_equal(store3.value, cse_var_2)


# -----------------------------------------------------------------------------------------
# A test which ensures that we don't perform normalizations outside of introduced variables
# -----------------------------------------------------------------------------------------
def test_no_normalization_without_commoning():
    x = te.var("x")
    y = te.var("y")
    z = te.var("z")
    a = te.var("a")
    # Test prog :
    # let a = x + (y + z) in a
    body = tvm.tir.LetStmt(a, x + (y + z), tvm.tir.Evaluate(a))

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([x, y, z], body))
    body = tvm.tir.transform.CommonSubexprElimTIR(identify_equiv_terms=True)(mod)

    tvm.transform.PrintIR()(body)

    body = body["main"].body  # Gets the body of the main, i.e. the full statement

    assert body.var.name == "a"
    assert tvm.ir.structural_equal(body.value, x + (y + z))


# -------------------------------------------------
# Part for testing the commoning with equivalences
# -------------------------------------------------
@T.prim_func
def func_distributivity(i1: T.int32, i2: T.int32, x: T.int32, y: T.int32, z: T.int32) -> None:
    B = T.Buffer((50,), "int32")
    B[i1] = x * (y + z)
    B[i2] = x * y + x * z


@T.prim_func
def func_distributivity_expected(
    i1: T.int32, i2: T.int32, x: T.int32, y: T.int32, z: T.int32
) -> None:
    B = T.Buffer((50,), "int32")
    with T.LetStmt(x * y + x * z) as cse_var_1:
        B[i1] = cse_var_1
        B[i2] = cse_var_1


@T.prim_func
def func_associativity(i1: T.int32, i2: T.int32, x: T.int32, y: T.int32, z: T.int32) -> None:
    B = T.Buffer((50,), "int32")
    B[i1] = (x + y) + z
    B[i2] = x + (y + z)


@T.prim_func
def func_associativity_expected(
    i1: T.int32, i2: T.int32, x: T.int32, y: T.int32, z: T.int32
) -> None:
    B = T.Buffer((50,), "int32")
    with T.LetStmt((x + y) + z) as cse_var_1:
        B[i1] = cse_var_1
        B[i2] = cse_var_1


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    body = tvm.tir.transform.CommonSubexprElimTIR(identify_equiv_terms=True)(mod)
    tvm.transform.PrintIR()(body)
    tvm.ir.assert_structural_equal(body["main"], transformed.with_attr("global_symbol", "main"))


def test_semantic_equiv_distributivity():
    _check(func_distributivity, func_distributivity_expected)


def test_semantic_equiv_associativity():
    _check(func_associativity, func_associativity_expected)


# -----------------------------------------------------
# Tests that verify the determinism of the pass
# -----------------------------------------------------
def test_deterministic_cse():
    import random

    """Test deterministic allocation of CSE vars

    We expect something like

        result = (x + 1) + (x + 2) + (x + 3) + (x + 1) + (x + 2) + (x + 3)
            -->
        cse_var_3 = (x + 1)
        cse_var_2 = (x + 2)
        cse_var_1 = (x + 3)
        result = cse_var_3 + cse_var_2 + cse_var_1 + cse_var_3 + cse_var_2 + cse_var_1
    """
    NUM_TERMS = 10
    REPEATS = 10

    x = te.var("x")
    result = te.var("result")

    offsets = sorted([i + 1 for i in range(NUM_TERMS)])
    inc1 = [(x + offsets[i]) for i in range(NUM_TERMS)]
    inc2 = [(x + offsets[i]) for i in range(NUM_TERMS)]

    expression = x
    for add in inc1 + inc2:
        expression = expression + add
    let_stmt = tvm.tir.LetStmt(result, expression, tvm.tir.Evaluate(result))
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([x], let_stmt))

    initial_hash = None
    for _ in range(REPEATS):
        body = tvm.tir.transform.CommonSubexprElimTIR()(mod)

        body = body["main"]

        # Hash and ensure serialize json is the same every time
        json_val = save_json(body)
        json_hash = hashlib.sha256(json_val.encode()).hexdigest()

        if initial_hash is None:
            initial_hash = json_hash
        assert json_hash == initial_hash


# Needed for the second test on determinism
LOG_LINE = '{"i": [["[\\"conv2d_layer\\", 1, 7, 7, 512, 512, 3, 3, [1, 1], [1, 1]]", \
            "llvm -keys=cpu -mcpu=broadwell -num-cores=2", \
            [8, 64, 64, 0, 0, 0, 0, 0], "", 1, []], [[], [["CI", 5], \
            ["SP", 3, 0, 1, [1, 1, 1], 1], ["SP", 3, 4, 512, [1, 32, 16], 1], \
            ["SP", 3, 8, 7, [7, 1, 1], 1], ["SP", 3, 12, 7, [1, 1, 1], 1], \
            ["SP", 3, 16, 512, [1], 1], ["SP", 3, 18, 3, [1], 1], ["SP", 3, 20, 3, [3], 1], \
            ["RE", 3, [0, 4, 8, 12, 1, 5, 9, 13, 16, 18, 20, 2, 6, 10, 14, 17, 19, 21, 3, 7, \
            11, 15]], ["FSP", 6, 0, 1, 2], ["FSP", 6, 3, 2, 2], ["FSP", 6, 6, 3, 2], \
            ["FSP", 6, 9, 4, 2], ["RE", 6, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]], \
            ["CA", 3, 6, 7], ["CA", 1, 6, 5], ["FU", 6, [0, 1, 2, 3, 4, 5]], ["AN", 6, 0, 3], \
            ["PR", 3, 0, "auto_unroll_max_step$512"], ["AN", 1, 3, 2], ["AN", 3, 21, 2], \
            ["AN", 6, 6, 2]]]], "r": [[0.0331129], 0, 0.900362, 1647464342], "v": "v0.6"}\n'

# The workload associated with the log
@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]


def test_deterministic_cse_2():
    inp, inr = auto_scheduler.measure_record.load_record_from_string(LOG_LINE)
    inp = auto_scheduler.measure.recover_measure_input(inp, rebuild_state=True)

    initial_hash = None

    for _ in range(10):
        sch, args = inp.task.compute_dag.apply_steps_from_state(inp.state)
        ir_module = tvm.lower(sch, args)
        primfunc = ir_module["main"]
        json_str = save_json(primfunc)
        new_hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
        # Make sure that all the hashes are going to be the same
        if initial_hash is None:
            initial_hash = new_hash
        assert new_hash == initial_hash


if __name__ == "__main__":
    # Basic test:
    test_cse()
    # Tests related to If nodes:
    test_cse_ifNode_1()
    test_cse_ifNode_2()
    # Test performing a commoning on a commoning:
    test_cse_cascade()
    # Test that verifies that the input program itself is not being normalized by the pass:
    test_no_normalization_without_commoning()
    # Tests that turn on the equivalence of terms and verify the commoning with equivalences:
    test_semantic_equiv_distributivity()
    test_semantic_equiv_associativity()
    # Tests that verify the determinism of the pass:
    test_deterministic_cse()
    test_deterministic_cse_2()
