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
# ruff: noqa: F401
import hashlib

import tvm
from tvm.ir.base import save_json
from tvm.ir.module import IRModule
from tvm.script import tir as T


# -----------------------------------------------------
# Basic test for the expected Behavior of the CSE pass
# -----------------------------------------------------
# A test program which gives the opportunity for the CSE pass to introduce two new variables,
# at two different levels
def test_cse():
    z1 = tvm.tir.Var("z1", "int32")
    z2 = tvm.tir.Var("z2", "int32")
    z3 = tvm.tir.Var("z3", "int32")
    i1 = tvm.tir.Var("i1", "int32")
    i2 = tvm.tir.Var("i2", "int32")
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    a = tvm.tir.Var("a", "int32")
    b = tvm.tir.Var("b", "int32")
    dtype = "int32"
    buffer = tvm.tir.decl_buffer((50,), dtype)
    # Test prog (flat Bind style):
    # z1 = 1; z2 = 2;
    # Mem[i1] = z1+z2;
    # x = 1; y = 1;
    # a = (x+y) + (z1+z2);
    # b = (x+y) + z3;
    # Mem[i2] = a+b;
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.Bind(z1, 1),
            tvm.tir.Bind(z2, 2),
            tvm.tir.BufferStore(buffer, z1 + z2, [i1]),
            tvm.tir.Bind(x, 1),
            tvm.tir.Bind(y, 1),
            tvm.tir.Bind(a, (x + y) + (z1 + z2)),
            tvm.tir.Bind(b, (x + y) + z3),
            tvm.tir.BufferStore(buffer, a + b, [i2]),
        ]
    )
    # This test program gives the opportunity to introduce two new variables,
    # and to perform replacements in the value of "a" and "b", using these new variables.

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([i1, i2, z3], body))
    body = tvm.tir.transform.CommonSubexprElimTIR()(mod)

    tvm.transform.PrintIR()(body)

    body = body["main"].body  # Gets the body of the main, i.e. the full statement

    # The result should be a flat SeqStmt with Bind nodes for z1, z2, cse_v1 (z1+z2),
    # the store, x, y, cse_v2 (x+y), a (using cse vars), b (using cse vars), store
    assert isinstance(body, tvm.tir.SeqStmt)

    # Walk through the flat sequence and check the CSE-introduced bindings
    stmts = list(body)
    idx = 0

    # z1 = 1
    assert isinstance(stmts[idx], tvm.tir.Bind)
    assert stmts[idx].var.name == "z1"
    assert stmts[idx].value == 1
    idx += 1

    # z2 = 2
    assert isinstance(stmts[idx], tvm.tir.Bind)
    assert stmts[idx].var.name == "z2"
    assert stmts[idx].value == 2
    idx += 1

    # CSE should introduce cse_v1 = z1 + z2 here
    assert isinstance(stmts[idx], tvm.tir.Bind)
    cse_v1 = stmts[idx].var
    assert stmts[idx].var.name == "cse_v1"
    tvm.ir.assert_structural_equal(stmts[idx].value, z1 + z2)
    idx += 1

    # Mem[i1] = cse_v1 (was z1+z2, now replaced)
    assert isinstance(stmts[idx], tvm.tir.BufferStore)
    tvm.ir.assert_structural_equal(stmts[idx].value, cse_v1)
    idx += 1

    # x = 1
    assert isinstance(stmts[idx], tvm.tir.Bind)
    assert stmts[idx].var.name == "x"
    assert stmts[idx].value == 1
    idx += 1

    # y = 1
    assert isinstance(stmts[idx], tvm.tir.Bind)
    assert stmts[idx].var.name == "y"
    assert stmts[idx].value == 1
    idx += 1

    # CSE should introduce cse_v2 = x + y here
    assert isinstance(stmts[idx], tvm.tir.Bind)
    cse_v2 = stmts[idx].var
    assert stmts[idx].var.name == "cse_v2"
    tvm.ir.assert_structural_equal(stmts[idx].value, x + y)
    idx += 1

    # a = cse_v2 + cse_v1 (was (x+y) + (z1+z2), now replaced)
    assert isinstance(stmts[idx], tvm.tir.Bind)
    assert stmts[idx].var.name == "a"
    tvm.ir.assert_structural_equal(stmts[idx].value, cse_v2 + cse_v1)
    idx += 1

    # b = cse_v2 + z3 (was (x+y) + z3, now replaced)
    assert isinstance(stmts[idx], tvm.tir.Bind)
    assert stmts[idx].var.name == "b"
    tvm.ir.assert_structural_equal(stmts[idx].value, cse_v2 + z3)
    idx += 1

    # Mem[i2] = a + b
    assert isinstance(stmts[idx], tvm.tir.BufferStore)
    idx += 1


# -----------------------------------------------------
# Tests related to If nodes
# -----------------------------------------------------
# First specific test for if nodes : Some duplicated computations appear only in one branch (here
# the Then branch), not in both branches.
# In this case, the CSE pass should introduce the redundant computation at the top of the Then
# branch, not before the whole If (otherwise that would lead to some computations being computed
# for nothing when it is the Else branch that is executed).
def test_cse_ifNode_1():
    b = tvm.tir.Var("b", "int32")
    i1 = tvm.tir.Var("i1", "int32")
    i2 = tvm.tir.Var("i2", "int32")
    i3 = tvm.tir.Var("i3", "int32")
    y = tvm.tir.Var("y", "int32")
    z = tvm.tir.Var("z", "int32")
    dtype = "int32"
    buffer = tvm.tir.decl_buffer((50,), dtype)
    # Test prog:
    # b = 1;
    # if(b) { Mem[i1] = y+z; Mem[i2] = y+z }
    # else { Mem[i3] = y }
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.Bind(b, 1),
            tvm.tir.IfThenElse(
                b,
                tvm.tir.SeqStmt(
                    [
                        tvm.tir.BufferStore(buffer, y + z, [i1]),
                        tvm.tir.BufferStore(buffer, y + z, [i2]),
                    ]
                ),
                tvm.tir.BufferStore(buffer, y, [i3]),
            ),
        ]
    )

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([i1, i2, i3, y, z], body))
    body = tvm.tir.transform.CommonSubexprElimTIR()(mod)

    tvm.transform.PrintIR()(body)

    body = body["main"].body  # Gets the body of the main, i.e. the full statement

    assert isinstance(body, tvm.tir.SeqStmt)
    stmts = list(body)

    # b = 1
    assert isinstance(stmts[0], tvm.tir.Bind)
    assert stmts[0].var.name == "b"
    assert stmts[0].value == 1

    # The If node
    assert isinstance(stmts[1], tvm.tir.IfThenElse)
    if_node = stmts[1]

    # The CSE variable should be inside the Then branch
    then_stmts = list(if_node.then_case)
    assert isinstance(then_stmts[0], tvm.tir.Bind)
    assert then_stmts[0].var.name == "cse_v1"
    tvm.ir.assert_structural_equal(then_stmts[0].value, y + z)


# Second test for if nodes : Some duplicated computations appear in both the Then and Else branch.
# In this case, the CSE pass should introduce the redundant computation before the whole If node,
# because regardless of the execution path, it is going to be computed.
def test_cse_ifNode_2():
    b = tvm.tir.Var("b", "int32")
    i1 = tvm.tir.Var("i1", "int32")
    i2 = tvm.tir.Var("i2", "int32")
    i3 = tvm.tir.Var("i3", "int32")
    y = tvm.tir.Var("y", "int32")
    z = tvm.tir.Var("z", "int32")
    dtype = "int32"
    buffer = tvm.tir.decl_buffer((50,), dtype)
    # Test prog:
    # b = 1;
    # if(b) { Mem[i1] = y+z; Mem[i2] = y }
    # else { Mem[i3] = y+z }
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.Bind(b, 1),
            tvm.tir.IfThenElse(
                b,
                tvm.tir.SeqStmt(
                    [
                        tvm.tir.BufferStore(buffer, y + z, [i1]),
                        tvm.tir.BufferStore(buffer, y, [i2]),
                    ]
                ),
                tvm.tir.BufferStore(buffer, y + z, [i3]),
            ),
        ]
    )

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([i1, i2, i3, y, z], body))
    body = tvm.tir.transform.CommonSubexprElimTIR()(mod)

    tvm.transform.PrintIR()(body)

    body = body["main"].body  # Gets the body of the main, i.e. the full statement

    assert isinstance(body, tvm.tir.SeqStmt)
    stmts = list(body)

    # CSE should introduce cse_v1 = y + z before the If
    # Find the cse_v1 binding
    found_cse = False
    for s in stmts:
        if isinstance(s, tvm.tir.Bind) and s.var.name == "cse_v1":
            tvm.ir.assert_structural_equal(s.value, y + z)
            found_cse = True
            break
    assert found_cse


# -------------------------------------------------------------------------------------------------
# Test commoning in cascade : after having introduced a big exp ((x+y)+z) into a new variable,
# it will become possible to do another commoning for (x+y) which appears both in the new variable
# and in the rest of the program.
# -------------------------------------------------------------------------------------------------
def test_cse_cascade():
    i1 = tvm.tir.Var("i1", "int32")
    i2 = tvm.tir.Var("i2", "int32")
    i3 = tvm.tir.Var("i3", "int32")
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    z = tvm.tir.Var("z", "int32")
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

    assert isinstance(body, tvm.tir.SeqStmt)
    stmts = list(body)

    # cse_v2 = x + y
    assert isinstance(stmts[0], tvm.tir.Bind)
    cse_v2 = stmts[0].var
    assert stmts[0].var.name == "cse_v2"
    tvm.ir.assert_structural_equal(stmts[0].value, (x + y))

    # cse_v1 = cse_v2 + z
    assert isinstance(stmts[1], tvm.tir.Bind)
    cse_v1 = stmts[1].var
    assert stmts[1].var.name == "cse_v1"
    tvm.ir.assert_structural_equal(stmts[1].value, cse_v2 + z)

    # Three stores
    assert isinstance(stmts[2], tvm.tir.BufferStore)
    assert isinstance(stmts[3], tvm.tir.BufferStore)
    assert isinstance(stmts[4], tvm.tir.BufferStore)

    tvm.ir.assert_structural_equal(stmts[2].value, cse_v1)
    tvm.ir.assert_structural_equal(stmts[3].value, cse_v1)
    tvm.ir.assert_structural_equal(stmts[4].value, cse_v2)


# -----------------------------------------------------------------------------------------
# A test which ensures that we don't perform normalizations outside of introduced variables
# -----------------------------------------------------------------------------------------
def test_no_normalization_without_commoning():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    z = tvm.tir.Var("z", "int32")
    a = tvm.tir.Var("a", "int32")
    # Test prog :
    # a = x + (y + z); evaluate(a)
    body = tvm.tir.SeqStmt([tvm.tir.Bind(a, x + (y + z)), tvm.tir.Evaluate(a)])

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([x, y, z], body))
    body = tvm.tir.transform.CommonSubexprElimTIR(identify_equiv_terms=True)(mod)

    tvm.transform.PrintIR()(body)

    body = body["main"].body  # Gets the body of the main, i.e. the full statement

    assert isinstance(body, tvm.tir.SeqStmt)
    stmts = list(body)
    assert isinstance(stmts[0], tvm.tir.Bind)
    assert stmts[0].var.name == "a"
    tvm.ir.assert_structural_equal(stmts[0].value, x + (y + z))


# -------------------------------------------------
# Part for testing the commoning with equivalences
# -------------------------------------------------
@T.prim_func
def func_distributivity(
    B: T.Buffer((50,), "int32"), i1: T.int32, i2: T.int32, x: T.int32, y: T.int32, z: T.int32
) -> None:
    B[i1] = (y + z) * x
    B[i2] = x * y + x * z


@T.prim_func
def func_distributivity_expected(
    B: T.Buffer((50,), "int32"), i1: T.int32, i2: T.int32, x: T.int32, y: T.int32, z: T.int32
) -> None:
    with T.LetStmt((y + z) * x) as cse_v1:
        B[i1] = cse_v1
        B[i2] = cse_v1


@T.prim_func
def func_associativity(
    B: T.Buffer((50,), "int32"), i1: T.int32, i2: T.int32, x: T.int32, y: T.int32, z: T.int32
) -> None:
    B[i1] = (x + y) + z
    B[i2] = x + (y + z)


@T.prim_func
def func_associativity_expected(
    B: T.Buffer((50,), "int32"), i1: T.int32, i2: T.int32, x: T.int32, y: T.int32, z: T.int32
) -> None:
    with T.LetStmt(x + y + z) as cse_v1:
        B[i1] = cse_v1
        B[i2] = cse_v1


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
        cse_v3 = (x + 1)
        cse_v2 = (x + 2)
        cse_v1 = (x + 3)
        result = cse_v3 + cse_v2 + cse_v1 + cse_v3 + cse_v2 + cse_v1
    """
    NUM_TERMS = 10
    REPEATS = 10

    x = tvm.tir.Var("x", "int32")
    result = tvm.tir.Var("result", "int32")

    offsets = sorted([i + 1 for i in range(NUM_TERMS)])
    inc1 = [(x + offsets[i]) for i in range(NUM_TERMS)]
    inc2 = [(x + offsets[i]) for i in range(NUM_TERMS)]

    expression = x
    for add in inc1 + inc2:
        expression = expression + add
    body = tvm.tir.SeqStmt([tvm.tir.Bind(result, expression), tvm.tir.Evaluate(result)])
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([x], body))

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
