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
from tvm.ir.base import save_json
from tvm.script import tir as T


# =====================================================================
# T1: Basic multi-level CSE
# Two common subexpressions at different scoping levels.
# =====================================================================
def test_basic():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), i1: T.int32, i2: T.int32, z3: T.int32):
            z1 = T.bind(1)
            z2 = T.bind(2)
            B[i1] = z1 + z2
            x = T.bind(1)
            y = T.bind(1)
            a = T.bind((x + y) + (z1 + z2))
            b = T.bind((x + y) + z3)
            B[i2] = a + b

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), i1: T.int32, i2: T.int32, z3: T.int32):
            z1 = T.bind(1)
            z2 = T.bind(2)
            cse_v1 = T.bind(z1 + z2)
            B[i1] = cse_v1
            x = T.bind(1)
            y = T.bind(1)
            cse_v2 = T.bind(x + y)
            a = T.bind(cse_v2 + cse_v1)
            b = T.bind(cse_v2 + z3)
            B[i2] = a + b

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T2: If -- single-branch CSE
# Duplicated expression only in then-branch stays inside then-branch.
# =====================================================================
def test_if_single_branch():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            i1: T.int32,
            i2: T.int32,
            i3: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            b = T.bind(1)
            if b:
                B[i1] = y + z
                B[i2] = y + z
            else:
                B[i3] = y

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            i1: T.int32,
            i2: T.int32,
            i3: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            b = T.bind(1)
            if b:
                cse_v1 = T.bind(y + z)
                B[i1] = cse_v1
                B[i2] = cse_v1
            else:
                B[i3] = y

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T3: If -- both-branch CSE
# Duplicated expression in both branches is hoisted before the if.
# =====================================================================
def test_if_both_branches():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            i1: T.int32,
            i2: T.int32,
            i3: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            b = T.bind(1)
            if b:
                B[i1] = y + z
                B[i2] = y
            else:
                B[i3] = y + z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            i1: T.int32,
            i2: T.int32,
            i3: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            b = T.bind(1)
            cse_v1 = T.bind(y + z)
            if b:
                B[i1] = cse_v1
                B[i2] = y
            else:
                B[i3] = cse_v1

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T4: Cascade CSE
# Introducing (x+y)+z creates opportunity for x+y.
# =====================================================================
def test_cascade():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            i1: T.int32,
            i2: T.int32,
            i3: T.int32,
            x: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            B[i1] = (x + y) + z
            B[i2] = (x + y) + z
            B[i3] = x + y

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            i1: T.int32,
            i2: T.int32,
            i3: T.int32,
            x: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            cse_v2 = T.bind(x + y)
            cse_v1 = T.bind(cse_v2 + z)
            B[i1] = cse_v1
            B[i2] = cse_v1
            B[i3] = cse_v2

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T5: No change when no duplication
# Single occurrence — pass is identity.
# =====================================================================
def test_no_duplication():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(x: T.int32, y: T.int32, z: T.int32):
            a = T.bind(x + (y + z))
            T.evaluate(a)

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(x: T.int32, y: T.int32, z: T.int32):
            a = T.bind(x + (y + z))
            T.evaluate(a)

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T6: Deterministic output
# Multiple runs on same input produce identical output (same var names,
# ordering). Verified via JSON serialization hash.
# =====================================================================
def test_deterministic():
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
        out = tvm.tir.transform.CommonSubexprElim()(mod)
        func = out["main"]
        json_val = save_json(func)
        json_hash = hashlib.sha256(json_val.encode()).hexdigest()
        if initial_hash is None:
            initial_hash = json_hash
        assert json_hash == initial_hash


# =====================================================================
# T7: CSE inside for-loop
# Common sub-expression inside a for-loop body.
# =====================================================================
def test_for_loop():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            for i in range(10):
                B[i] = y + z
                B[i + 10] = y + z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            for i in range(10):
                cse_v1 = T.bind(y + z)
                B[i] = cse_v1
                B[i + 10] = cse_v1

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T8: CSE across for-loop and outer scope
# Expression appears both outside and inside a for-loop. Binding placed
# in outer scope before the for.
# =====================================================================
def test_for_hoist():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            B[0] = y + z
            for i in range(10):
                B[i + 1] = y + z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            cse_v1 = T.bind(y + z)
            B[0] = cse_v1
            for i in range(10):
                B[i + 1] = cse_v1

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T9: Cannot-lift -- expressions containing BufferLoad
# Expressions containing BufferLoad are ineligible even when duplicated,
# because lifting them would change semantics (buffer may alias).
# =====================================================================
def test_cannot_lift_bufferload():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((50,), "int32"), B: T.Buffer((50,), "int32")):
            B[0] = A[0] + A[0]
            B[1] = A[0] + A[0]

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer((50,), "int32"), B: T.Buffer((50,), "int32")):
            B[0] = A[0] + A[0]
            B[1] = A[0] + A[0]

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T10: Nested if -- multi-level scope LCA
# Expression in both branches of an inner if, but not the outer else.
# Binding placed at the inner if's parent scope.
# =====================================================================
def test_nested_if():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            c1: T.int32,
            c2: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            if c1:
                if c2:
                    B[0] = y + z
                else:
                    B[1] = y + z
            else:
                B[2] = y

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            c1: T.int32,
            c2: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            if c1:
                cse_v1 = T.bind(y + z)
                if c2:
                    B[0] = cse_v1
                else:
                    B[1] = cse_v1
            else:
                B[2] = y

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T11: Multiple independent CSE candidates
# Several independent expressions, each duplicated.
# =====================================================================
def test_multi_independent():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            a: T.int32,
            b: T.int32,
            c: T.int32,
            d: T.int32,
        ):
            B[0] = a + b
            B[1] = c + d
            B[2] = a + b
            B[3] = c + d

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            a: T.int32,
            b: T.int32,
            c: T.int32,
            d: T.int32,
        ):
            cse_v1 = T.bind(a + b)
            B[0] = cse_v1
            cse_v2 = T.bind(c + d)
            B[1] = cse_v2
            B[2] = cse_v1
            B[3] = cse_v2

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T12: Expression in if-condition and branch
# One occurrence in the if-condition (parent scope), one in the
# then-branch. Binding hoisted before the if.
# =====================================================================
def test_if_condition():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            if y + z > 0:
                B[0] = y + z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            cse_v1 = T.bind(y + z)
            if cse_v1 > 0:
                B[0] = cse_v1

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T13: Cannot-lift -- expression containing Call
# Function calls cannot be lifted.
# =====================================================================
def test_cannot_lift_call():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), x: T.int32):
            B[0] = T.call_extern("my_func", x, dtype="int32") + 1
            B[1] = T.call_extern("my_func", x, dtype="int32") + 1

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), x: T.int32):
            B[0] = T.call_extern("my_func", x, dtype="int32") + 1
            B[1] = T.call_extern("my_func", x, dtype="int32") + 1

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T14: No single-use binding
# When all occurrences of a sub-expression are inside a deeper
# expression that is also CSE'd, the sub-expression should NOT get
# its own binding (it would be used only once in the parent's value).
# =====================================================================
def test_no_single_use_binding():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            x: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            B[0] = (x + y) + z
            B[1] = (x + y) + z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            x: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            cse_v1 = T.bind((x + y) + z)
            B[0] = cse_v1
            B[1] = cse_v1

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T15: For-loop extent lifting
# Expression shared between loop extent and body is hoisted outside the
# loop rather than being re-evaluated every iteration.
# =====================================================================
def test_for_extent_lift():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            for i in range(y + z):
                B[i] = y + z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            cse_v1 = T.bind(y + z)
            for i in range(cse_v1):
                B[i] = cse_v1

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T16: Loop-variable expression stays inside loop
# An expression using the loop variable (i*4) that appears multiple
# times inside the body must NOT be hoisted outside the loop.
# =====================================================================
def test_loop_var_expr_stays_inside():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
            A: T.Buffer((50,), "int32"),
            B: T.Buffer((50,), "int32"),
        ):
            for i in range(10):
                A[i * 4] = B[i * 4]

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(
            A: T.Buffer((50,), "int32"),
            B: T.Buffer((50,), "int32"),
        ):
            for i in range(10):
                cse_v1 = T.bind(i * 4)
                A[cse_v1] = B[cse_v1]

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T17: No normalization without commoning
# Single-occurrence expression is not normalized or modified.
# =====================================================================
def test_no_normalization_without_commoning():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(x: T.int32, y: T.int32, z: T.int32):
            a = T.bind(x + (y + z))
            T.evaluate(a)

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(x: T.int32, y: T.int32, z: T.int32):
            a = T.bind(x + (y + z))
            T.evaluate(a)

    after = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


# =====================================================================
# T18: Let-bound variable -- no extraction from Let body
# Expressions inside a Let body that reference the Let-bound variable
# must NOT be extracted, because the variable is only in scope inside
# the Let body, not at the statement level where Bind would be placed.
# =====================================================================
def test_let_body_no_extraction():
    """CSE must not extract expressions from Let bodies that use Let-bound vars."""
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    # Let(x, 1, (x+y) + (x+y)) -- x+y appears twice but x is Let-bound
    let_expr = tvm.tir.Let(x, tvm.tir.IntImm("int32", 1), (x + y) + (x + y))
    buf = tvm.tir.decl_buffer((10,), "int32", name="B")
    i = tvm.tir.Var("i", "int32")
    store = tvm.tir.BufferStore(buf, let_expr, [i])
    loop = tvm.tir.For(
        i,
        tvm.tir.const(0, "int32"),
        tvm.tir.const(10, "int32"),
        tvm.tir.ForKind.SERIAL,
        store,
    )
    func = tvm.tir.PrimFunc([buf, y], loop)
    mod = tvm.IRModule({"main": func})
    mod_after = tvm.tir.transform.CommonSubexprElim()(mod)
    # No CSE variables should be introduced
    script = mod_after["main"].script()
    assert "cse_v" not in script, f"CSE incorrectly extracted from Let body:\n{script}"


# =====================================================================
# T19: Let value -- CSE works for shared Let value expressions
# The Let value is evaluated before the binding takes effect, so
# expressions in the Let value CAN be CSE'd with expressions outside.
# =====================================================================
def test_let_value_cse():
    """CSE can extract from Let values (computed before binding)."""
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    z = tvm.tir.Var("z", "int32")
    # Let(x, y+z, x+1) with y+z also appearing outside the Let
    let_expr = tvm.tir.Let(x, y + z, x + 1)
    buf = tvm.tir.decl_buffer((10,), "int32", name="B")
    i = tvm.tir.Var("i", "int32")
    store = tvm.tir.BufferStore(buf, (y + z) + let_expr, [i])
    loop = tvm.tir.For(
        i,
        tvm.tir.const(0, "int32"),
        tvm.tir.const(10, "int32"),
        tvm.tir.ForKind.SERIAL,
        store,
    )
    func = tvm.tir.PrimFunc([buf, y, z], loop)
    mod = tvm.IRModule({"main": func})
    mod_after = tvm.tir.transform.CommonSubexprElim()(mod)
    # y+z should be extracted (appears in Let value AND outside)
    script = mod_after["main"].script()
    assert "cse_v" in script, f"CSE should extract y+z from Let value:\n{script}"


# =====================================================================
# T20: Nested Let -- both levels respected
# Multiple nested Let expressions with common sub-expressions in the
# innermost body. None should be extracted.
# =====================================================================
def test_nested_let_no_extraction():
    """CSE must not extract from nested Let bodies."""
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    z = tvm.tir.Var("z", "int32")
    # Let(x, 1, Let(y, 2, (x+y+z) + (x+y+z)))
    inner = (x + y + z) + (x + y + z)
    nested_let = tvm.tir.Let(
        x, tvm.tir.IntImm("int32", 1), tvm.tir.Let(y, tvm.tir.IntImm("int32", 2), inner)
    )
    buf = tvm.tir.decl_buffer((10,), "int32", name="B")
    i = tvm.tir.Var("i", "int32")
    store = tvm.tir.BufferStore(buf, nested_let, [i])
    loop = tvm.tir.For(
        i,
        tvm.tir.const(0, "int32"),
        tvm.tir.const(10, "int32"),
        tvm.tir.ForKind.SERIAL,
        store,
    )
    func = tvm.tir.PrimFunc([buf, z], loop)
    mod = tvm.IRModule({"main": func})
    mod_after = tvm.tir.transform.CommonSubexprElim()(mod)
    script = mod_after["main"].script()
    assert "cse_v" not in script, f"CSE incorrectly extracted from nested Let body:\n{script}"


# =====================================================================
# T21: Let with lowered floordiv pattern
# Simulates the pattern produced by LowerIntrin for floordiv:
# Let(rmod, truncmod(a,b), Let(rdiv, div(a,b), Select(...)))
# wrapped in Let(x, load, Let(y, load, ...))
# This is the regression test for the CI failure in lower_intrin tests.
# =====================================================================
def test_let_floordiv_pattern():
    """CSE must handle the Let pattern from LowerIntrin's floordiv lowering."""
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    rmod = tvm.tir.Var("rmod", "int32")
    rdiv = tvm.tir.Var("rdiv", "int32")
    # Simulate lowered floordiv: Let(rmod, x%y, Let(rdiv, x/y, Select(...)))
    select_cond = tvm.tir.And(y >= 0, rmod >= 0) | tvm.tir.And(y < 0, rmod <= 0)
    select_expr = tvm.tir.Select(select_cond, rdiv, rdiv - 1)
    inner_let = tvm.tir.Let(rdiv, tvm.tir.Div(x, y), select_expr)
    outer_let = tvm.tir.Let(rmod, tvm.tir.Mod(x, y), inner_let)
    # Wrap in Let(x, load, Let(y, load, ...))
    buf_a = tvm.tir.decl_buffer((10,), "int32", name="A")
    buf_b = tvm.tir.decl_buffer((10,), "int32", name="B")
    buf_c = tvm.tir.decl_buffer((10,), "int32", name="C")
    i = tvm.tir.Var("i", "int32")
    full_expr = tvm.tir.Let(
        x,
        tvm.tir.BufferLoad(buf_a, [i]),
        tvm.tir.Let(y, tvm.tir.BufferLoad(buf_b, [i]), outer_let),
    )
    store = tvm.tir.BufferStore(buf_c, full_expr, [i])
    loop = tvm.tir.For(
        i,
        tvm.tir.const(0, "int32"),
        tvm.tir.const(10, "int32"),
        tvm.tir.ForKind.SERIAL,
        store,
    )
    func = tvm.tir.PrimFunc([buf_a, buf_b, buf_c], loop)
    mod = tvm.IRModule({"main": func})
    # Should not crash and should not extract Let-bound vars
    mod_after = tvm.tir.transform.CommonSubexprElim()(mod)
    script = mod_after["main"].script()
    assert "cse_v" not in script, f"CSE incorrectly extracted from Let body:\n{script}"


if __name__ == "__main__":
    test_basic()
    test_if_single_branch()
    test_if_both_branches()
    test_cascade()
    test_no_duplication()
    test_deterministic()
    test_for_loop()
    test_for_hoist()
    test_cannot_lift_bufferload()
    test_nested_if()
    test_multi_independent()
    test_if_condition()
    test_cannot_lift_call()
    test_no_single_use_binding()
    test_for_extent_lift()
    test_loop_var_expr_stays_inside()
    test_no_normalization_without_commoning()
    test_let_body_no_extraction()
    test_let_value_cse()
    test_nested_let_no_extraction()
    test_let_floordiv_pattern()
