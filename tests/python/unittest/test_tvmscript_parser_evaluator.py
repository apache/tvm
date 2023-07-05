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
"""Unittests for tvm.script.parser.evaluator"""
import pytest
import tvm.testing
from tvm.script.parser.core.diagnostics import Source
from tvm.script.parser.core.evaluator import ExprEvaluator


def _calc(expr, extra_vars=None):
    if extra_vars is None:
        extra_vars = {}
    source = Source(expr)
    mod_ast = source.as_ast()
    mod_body_ast = mod_ast.body
    expr_stmt_ast = mod_body_ast[0]
    expr_ast = expr_stmt_ast.value
    return ExprEvaluator.eval(None, extra_vars, expr_ast)


def test_evaluator_basic():
    assert _calc("1, 3.14, True, 'str'") == (1, 3.14, True, "str")


def test_evaluator_op():
    assert _calc("1 + 2, 1 - 2, 1 * 2, 1 / 2") == (3, -1, 2, 0.5)


def test_evaluator_value_table():
    res = _calc("a + b, a - b, a * b, a / b", {"a": 1, "b": 2})
    a, b = 1, 2
    assert res == (a + b, a - b, a * b, a / b)


def test_evaluator_func_call():
    def func(a, b):
        return a + b, a - b, a * b, a / b

    assert _calc("func(1, 2)", {"func": func}) == func(1, 2)


def test_evaluator_slice():
    res = _calc("a, a[1:], a[:5], a[1: 5], a[1: 5: 2]", {"a": [1, 2, 3, 4, 5, 6]})
    a = [1, 2, 3, 4, 5, 6]
    assert res == (a, a[1:], a[:5], a[1:5], a[1:5:2])


if __name__ == "__main__":
    tvm.testing.main()
