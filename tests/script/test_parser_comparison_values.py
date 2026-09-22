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
"""Source equality reaches typed consumers as concrete boolean IR."""

import pytest
import tvm_ffi

import tvm
from tvm import ir, tirx
from tvm.ir._overload_prim_expr import EqualOp
from tvm.script import parser
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder


@pytest.mark.parametrize("operator, kind", [("==", tirx.EQ), ("!=", tirx.NE)])
@pytest.mark.parametrize("track_span", [True, False])
def test_symbolic_equality_reaches_typed_consumer(operator, kind, track_span):
    seen = []

    def consume(value):
        assert isinstance(value, kind)
        assert str(value.ty) == "bool"
        seen.append(value)
        return value

    source = f"""
@T.prim_func
def main(x: T.int32):
    T.evaluate(consume(x {operator} 0))
"""
    actual = parser.parse(source, extra_vars={"consume": consume}, track_span=track_span)
    assert len(seen) == 1
    assert seen[0].a.same_as(actual.params[0])
    if track_span:
        assert seen[0].span is not None


def test_python_comparisons_remain_python_booleans():
    seen = []

    def consume(value):
        assert type(value) is bool
        seen.append(value)
        return int(value)

    parser.parse(
        """
@T.prim_func
def main():
    T.evaluate(consume(T.constexpr([1, 2] == [1, 2])))
    T.evaluate(consume(T.constexpr("x" != "y")))
""",
        extra_vars={"consume": consume},
    )
    assert seen == [True, True]


def test_host_equality_result_is_not_arbitrarily_converted():
    class HostResult(tvm_ffi.ObjectConvertible):
        def asobject(self):
            raise AssertionError("host comparison result must not be converted")

    result = HostResult()

    class Host:
        def __eq__(self, other):
            return result

    seen = []

    def consume(value):
        assert value is result
        seen.append(value)
        return 0

    parser.parse(
        """
@T.prim_func
def main():
    T.evaluate(consume(T.constexpr(left == right)))
""",
        extra_vars={"left": Host(), "right": Host(), "consume": consume},
    )
    assert seen == [result]
    assert seen[0] is result

    class CustomEquality(EqualOp):
        def asobject(self):
            raise AssertionError("custom equality subclasses are host values")

    custom = CustomEquality(tirx.Var("x", "int32"), 0)
    result = custom
    parser.parse(
        """
@T.prim_func
def main():
    T.evaluate(consume(T.constexpr(left == right)))
""",
        extra_vars={"left": Host(), "right": Host(), "consume": consume},
    )
    assert seen[-1] is custom


@pytest.mark.parametrize(
    "expression", ["operand(0) == operand(1)", "operand(0) == operand(1) != operand(2)"]
)
def test_comparison_operands_evaluate_once_in_order(expression):
    seen = []

    def operand(index, value):
        seen.append(index)
        return value

    def consume(value):
        assert isinstance(value, ir.Expr)
        assert str(value.ty) == "bool"
        return value

    expression = expression.replace("operand(0)", "operand(0, value0)")
    expression = expression.replace("operand(1)", "operand(1, value1)")
    expression = expression.replace("operand(2)", "operand(2, value2)")
    parser.parse(
        "@T.prim_func\ndef main(value0: T.int32, value1: T.int32, value2: T.int32):\n"
        f"    T.evaluate(consume({expression}))\n",
        extra_vars={"operand": operand, "consume": consume},
    )
    assert seen == ([0, 1, 2] if "!=" in expression else [0, 1])


@pytest.mark.parametrize(
    "kind,source",
    [
        (
            "store",
            """
@T.prim_func
def main(dst: T.Buffer((2,), "uint32")):
    T.device_entry()
    tx = T.thread_id([32])
    T.ptx.st.global_.v2.b32(
        T.ptx.addr(dst.data, 0), T.uint32(1), T.uint32(2), pred=tx == 0
    )
""",
        ),
        (
            "mma",
            """
@T.prim_func
def main():
    T.device_entry()
    tx = T.thread_id([32])
    tmem = T.local_scalar("uint32")
    desc = T.local_scalar("uint64")
    idesc = T.local_scalar("uint32")
    T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
        tmem, desc, desc, idesc, 0, 0, 0, 0, tx == 0
    )
""",
        ),
    ],
)
def test_backend_predicate_materializes_equality(kind, source):
    with IRBuilder() as builder:
        with T.function():
            T.func_name("main")
            if kind == "store":
                dst = T.arg("dst", T.Buffer((2,), "uint32"))
            T.device_entry()
            tx = T.thread_id([32]).value
            predicate = tirx.EQ(tx, 0)
            if kind == "store":
                T.emit_(
                    T.ptx.st.global_.v2.b32(
                        T.ptx.addr(dst.data, 0), T.uint32(1), T.uint32(2), pred=predicate
                    )
                )
            else:
                tmem = T.local_scalar("uint32")
                desc = T.local_scalar("uint64")
                idesc = T.local_scalar("uint32")
                T.emit_(
                    T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
                        tmem, desc, desc, idesc, 0, 0, 0, 0, predicate
                    )
                )
    expected = builder.get()
    actual = parser.parse(source)
    tvm.ir.assert_structural_equal(expected, actual)
