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
import tvm.testing
from tvm.ir.base import save_json
from tvm.script import tirx as T


def test_basic():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((50,), "int32"), i1: T.int32, i2: T.int32, z3: T.int32):
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
        @T.function
        def main(B: T.Tensor((50,), "int32"), i1: T.int32, i2: T.int32, z3: T.int32):
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

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_if_single_branch():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(
            B: T.Tensor((50,), "int32"),
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
        @T.function
        def main(
            B: T.Tensor((50,), "int32"),
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

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_if_both_branches():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(
            B: T.Tensor((50,), "int32"),
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
        @T.function
        def main(
            B: T.Tensor((50,), "int32"),
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

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_cascade():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(
            B: T.Tensor((50,), "int32"),
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
        @T.function
        def main(
            B: T.Tensor((50,), "int32"),
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

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_no_duplication():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(x: T.int32, y: T.int32, z: T.int32):
            a = T.bind(x + (y + z))
            T.evaluate(a)

    Expected = Before

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_deterministic():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((4,), "int32"), x: T.int32):
            B[0] = x + 1
            B[1] = x + 2
            B[2] = x + 1
            B[3] = x + 2

    @tvm.script.ir_module
    class Expected:
        @T.function
        def main(B: T.Tensor((4,), "int32"), x: T.int32):
            cse_v1 = T.bind(x + 1)
            B[0] = cse_v1
            cse_v2 = T.bind(x + 2)
            B[1] = cse_v2
            B[2] = cse_v1
            B[3] = cse_v2

    hashes = set()
    for _ in range(10):
        After = tvm.tirx.transform.CommonSubexprElim()(Before)
        tvm.ir.assert_structural_equal(After, Expected)
        hashes.add(hashlib.sha256(save_json(After).encode()).hexdigest())
    assert len(hashes) == 1


def test_for_loop():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((50,), "int32"), y: T.int32, z: T.int32):
            for i in range(10):
                B[i] = y + z
                B[i + 10] = y + z

    @tvm.script.ir_module
    class Expected:
        @T.function
        def main(B: T.Tensor((50,), "int32"), y: T.int32, z: T.int32):
            for i in range(10):
                cse_v1 = T.bind(y + z)
                B[i] = cse_v1
                B[i + 10] = cse_v1

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_for_hoist():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((50,), "int32"), y: T.int32, z: T.int32):
            B[0] = y + z
            for i in range(10):
                B[i + 1] = y + z

    @tvm.script.ir_module
    class Expected:
        @T.function
        def main(B: T.Tensor((50,), "int32"), y: T.int32, z: T.int32):
            cse_v1 = T.bind(y + z)
            B[0] = cse_v1
            for i in range(10):
                B[i + 1] = cse_v1

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_cannot_lift_tensorload():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((50,), "int32"), B: T.Tensor((50,), "int32")):
            B[0] = A[0] + A[0]
            B[1] = A[0] + A[0]

    Expected = Before

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_nested_if():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(
            B: T.Tensor((50,), "int32"),
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
        @T.function
        def main(
            B: T.Tensor((50,), "int32"),
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

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_multi_independent():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(
            B: T.Tensor((50,), "int32"),
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
        @T.function
        def main(
            B: T.Tensor((50,), "int32"),
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

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_if_condition():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((50,), "int32"), y: T.int32, z: T.int32):
            if y + z > 0:
                B[0] = y + z

    @tvm.script.ir_module
    class Expected:
        @T.function
        def main(B: T.Tensor((50,), "int32"), y: T.int32, z: T.int32):
            cse_v1 = T.bind(y + z)
            if cse_v1 > 0:
                B[0] = cse_v1

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_cannot_lift_call():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((50,), "int32"), x: T.int32):
            B[0] = T.call_extern("my_func", x, ty="int32") + 1
            B[1] = T.call_extern("my_func", x, ty="int32") + 1

    Expected = Before

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_no_single_use_binding():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(
            B: T.Tensor((50,), "int32"),
            x: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            B[0] = (x + y) + z
            B[1] = (x + y) + z

    @tvm.script.ir_module
    class Expected:
        @T.function
        def main(
            B: T.Tensor((50,), "int32"),
            x: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            cse_v1 = T.bind((x + y) + z)
            B[0] = cse_v1
            B[1] = cse_v1

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_for_extent_lift():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((50,), "int32"), y: T.int32, z: T.int32):
            for i in range(y + z):
                B[i] = y + z

    @tvm.script.ir_module
    class Expected:
        @T.function
        def main(B: T.Tensor((50,), "int32"), y: T.int32, z: T.int32):
            cse_v1 = T.bind(y + z)
            for i in range(cse_v1):
                B[i] = cse_v1

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_loop_var_expr_stays_inside():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(
            A: T.Tensor((50,), "int32"),
            B: T.Tensor((50,), "int32"),
        ):
            for i in range(10):
                A[i * 4] = B[i * 4]

    @tvm.script.ir_module
    class Expected:
        @T.function
        def main(
            A: T.Tensor((50,), "int32"),
            B: T.Tensor((50,), "int32"),
        ):
            for i in range(10):
                cse_v1 = T.bind(i * 4)
                A[cse_v1] = B[cse_v1]

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_let_body_no_extraction():
    x = T.dynamic("x", "int32")

    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((10,), "int32"), y: T.int32):
            for i in range(10):
                B[i] = T.Let((x + y) + (x + y), where={x: 1})

    After = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(After, Before)


def test_let_value_cse():
    x = T.dynamic("x", "int32")

    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((10,), "int32"), y: T.int32, z: T.int32):
            for i in range(10):
                B[i] = (y + z) + T.Let(x + 1, where={x: y + z})

    @tvm.script.ir_module
    class Expected:
        @T.function
        def main(B: T.Tensor((10,), "int32"), y: T.int32, z: T.int32):
            for i in range(10):
                cse_v1 = T.bind(y + z)
                B[i] = cse_v1 + T.Let(x + 1, where={x: cse_v1})

    After = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_nested_let_no_extraction():
    x = T.dynamic("x", "int32")
    y = T.dynamic("y", "int32")

    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((10,), "int32"), z: T.int32):
            for i in range(10):
                B[i] = T.Let(T.Let((x + y + z) + (x + y + z), where={y: 2}), where={x: 1})

    After = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(After, Before)


def test_let_floordiv_pattern():
    x = T.dynamic("x", "int32")
    y = T.dynamic("y", "int32")
    rmod = T.dynamic("rmod", "int32")
    rdiv = T.dynamic("rdiv", "int32")

    @tvm.script.ir_module
    class Before:
        @T.function
        def main(
            A: T.Tensor((10,), "int32"), B: T.Tensor((10,), "int32"), C: T.Tensor((10,), "int32")
        ):
            for i in range(10):
                C[i] = T.Let(
                    T.Let(
                        T.Let(
                            T.Let(
                                T.Select(
                                    (y >= 0 and rmod >= 0) or (y < 0 and rmod <= 0), rdiv, rdiv - 1
                                ),
                                where={rdiv: T.Div(x, y)},
                            ),
                            where={rmod: T.Mod(x, y)},
                        ),
                        where={y: B[i]},
                    ),
                    where={x: A[i]},
                )

    After = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(After, Before)


def test_no_lift_bool_predicate():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((50,), "int32"), n: T.int32, x: T.int32):
            for i in range(50):
                if i < n:
                    B[i] = x
                if i < n:
                    B[i] = x + 1

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Before)
    assert "cse_v" not in after["main"].script()


def test_no_lift_bool_logical():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((50,), "int32"), a: T.bool, b: T.bool, x: T.int32):
            if T.And(a, b):
                B[0] = x
            if T.And(a, b):
                B[1] = x + 1

    after = tvm.tirx.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(after, Before)
    assert "cse_v" not in after["main"].script()


# T24: Shared subtree stays SSA. UnrollLoop reuses one Stmt object across
# positions; the identity-keyed rewriter must bind fresh cse vars per occurrence.
def test_shared_subtree_stays_ssa():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((50,), "int32"), i1: T.int32, i2: T.int32):
            for unused in T.unroll(2):
                B[i1] = (i1 + i2) * 2
                B[i2] = (i1 + i2) * 3

    @tvm.script.ir_module
    class Expected:
        @T.function
        def main(B: T.Tensor((50,), "int32"), i1: T.int32, i2: T.int32):
            value = T.bind(i1 + i2)
            twice = T.bind(value * 2)
            B[i1] = twice
            thrice = T.bind(value * 3)
            B[i2] = thrice
            value2 = T.bind(i1 + i2)
            twice2 = T.bind(value2 * 2)
            B[i1] = twice2
            thrice2 = T.bind(value * 3)
            B[i2] = thrice2

    unrolled = tvm.tirx.transform.UnrollLoop()(Before)
    body = unrolled["main"].body
    assert body[0].same_as(body[2])
    assert body[1].same_as(body[3])
    After = tvm.tirx.transform.CommonSubexprElim()(unrolled)
    tvm.ir.assert_structural_equal(After, Expected)
    assert tvm.tirx.analysis.verify_well_formed(After["main"])


if __name__ == "__main__":
    tvm.testing.main()
