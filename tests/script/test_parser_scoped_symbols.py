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
"""Signature symbols cross call scopes without changing introduction order."""

import pytest

from tvm.error import DiagnosticError
from tvm.script import parser


@pytest.mark.parametrize("track_span", [True, False])
def test_signature_symbols_cross_nested_calls_parameters_return_and_body(track_span):
    function = parser.parse(
        """
@R.function
def main(
    x: R.Tuple(R.Tensor(("n",), "float32"), R.Tensor((n,), "float32")),
    y: R.Tensor((n,), "float32"),
) -> R.Tensor((n,), "float32"):
    R.func_attr({"symbol": n})
    return y
""",
        track_span=track_span,
    )
    n = function.params[0].ty.fields[0].shape[0]
    assert function.params[0].ty.fields[1].shape[0].same_as(n)
    assert function.params[1].ty.shape[0].same_as(n)
    assert function.ret_ty.shape[0].same_as(n)
    assert function.attrs["symbol"].same_as(n)


def test_signature_string_binds_before_later_dimension():
    function = parser.parse(
        """
@R.function
def main(x: R.Tensor(("n + 1", n), "float32")):
    R.func_attr({"symbol": n})
    return x
"""
    )
    first, n = function.params[0].ty.shape
    assert first.a.same_as(n)
    assert first.b.value == 1
    assert function.attrs["symbol"].same_as(n)


@pytest.mark.parametrize(
    "annotation",
    [
        'R.Tensor((n, "n"), "float32")',
        'R.Tuple(R.Tensor((n,), "float32"), R.Tensor(("n",), "float32"))',
    ],
)
def test_signature_read_before_introduction_remains_unbound(annotation):
    source = f"""
@R.function
def main(x: {annotation}):
    return x
"""
    with pytest.raises(DiagnosticError) as error:
        parser.parse(source)
    assert isinstance(error.value.__cause__, NameError)
    assert "n" in str(error.value.__cause__)


def test_signature_capture_precedes_string_rebinding():
    function = parser.parse(
        """
@R.function
def main(x: R.Tensor((n, "n"), "float32"), y: R.Tensor((n,), "float32")):
    R.func_attr({"symbol": n})
    return y
""",
        extra_vars={"n": 7},
    )
    first, n = function.params[0].ty.shape
    assert first.value == 7
    assert function.params[1].ty.shape[0].same_as(n)
    assert function.attrs["symbol"].same_as(n)


def test_static_resolution_preserves_instance_functions_and_bound_methods():
    import ast
    import inspect
    from types import SimpleNamespace

    from tvm.script.parser.transpile import IRBuilderTranspiler

    class Owner:
        def constructor(self, shape):
            return shape

    instance = Owner()
    namespace = SimpleNamespace(constructor=Owner.constructor)
    shadowed = Owner()
    shadowed.constructor = Owner.constructor
    transformer = IRBuilderTranspiler(
        "<static-resolution>",
        {"instance": instance, "namespace": namespace, "shadowed": shadowed},
        "X",
        "I",
        lambda node: ast.Constant(None),
        None,
    )
    for name in ("namespace", "shadowed"):
        resolved = transformer._resolve(ast.parse(f"{name}.constructor", mode="eval").body)
        assert resolved is Owner.constructor
        assert list(inspect.signature(resolved).parameters) == ["self", "shape"]
    resolved = transformer._resolve(ast.parse("instance.constructor", mode="eval").body)
    assert resolved.__self__ is instance
    assert list(inspect.signature(resolved).parameters) == ["shape"]
