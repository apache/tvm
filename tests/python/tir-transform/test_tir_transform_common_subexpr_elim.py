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
