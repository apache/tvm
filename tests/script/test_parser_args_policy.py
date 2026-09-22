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
"""Per-argument expression and module-reference syntax policies."""

import ast
from types import SimpleNamespace
from typing import TypeVar

import pytest

from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder.type_var_frame import TypeVarFrame
from tvm.script.parser import protocol
from tvm.script.parser.expression import rewrite_expression


def _rewrite(source, environment, **kwargs):
    node = ast.parse(source, mode="eval").body

    def resolve(value):
        if isinstance(value, ast.Name):
            return environment.get(value.id)
        if isinstance(value, ast.Attribute):
            return getattr(resolve(value.value), value.attr, None)
        return None

    result = rewrite_expression(node, resolve, "X", "<args-policy>", **kwargs)
    return node, result


def _evaluate(node, environment):
    return eval(compile(ast.Expression(node), "<args-policy>", "eval"), environment)


@pytest.mark.parametrize(
    "source",
    [
        'tensor(("n + 1",), "float32", "cuda:1")',
        'tensor(shape=("n + 1",), dtype="float32", vdevice="cuda:1")',
    ],
)
def test_expression_and_global_info_arguments(source):
    @protocol.args_policy({"shape": "expr_str", "vdevice": "global_info"}, scalar_strings=False)
    def tensor(shape, dtype, vdevice):
        return shape, dtype, vdevice

    seen = []
    device = object()

    def lookup(content):
        seen.append(content)
        return device

    env = {
        "tensor": tensor,
        "X": SimpleNamespace(resolve_type_var=lambda name: {"n": 7}[name]),
        "_PS_1": SimpleNamespace(lookup_global_info=lookup),
    }
    original, rewritten = _rewrite(source, env, parser_support="_PS_1")
    assert ast.dump(original) == ast.dump(ast.parse(source, mode="eval").body)
    assert _evaluate(rewritten, env) == ((8,), "float32", device)
    assert seen == ["cuda:1"]
    generated = [
        node for node in ast.walk(rewritten) if getattr(node, "_tvm_parser_support", False)
    ]
    assert len(generated) == 2


def test_mesh_reference_and_placement_are_not_expression_strings():
    @protocol.args_policy({"shape": "expr_str", "device_mesh": "global_info"})
    def distributed(shape, dtype, device_mesh, placement):
        return shape, dtype, device_mesh, placement

    mesh = object()
    references = []

    def lookup(content):
        references.append(content)
        return mesh

    env = {
        "distributed": distributed,
        "X": SimpleNamespace(resolve_type_var=lambda name: 16),
        "_PS": SimpleNamespace(lookup_global_info=lookup),
    }
    _, rewritten = _rewrite('distributed(["n"], "float32", "mesh[0]", "S[0]")', env)
    assert _evaluate(rewritten, env) == ([16], "float32", mesh, "S[0]")
    assert references == ["mesh[0]"]


def test_global_info_content_is_evaluated_once_in_argument_order():
    @protocol.args_policy({"device": "global_info"})
    def construct(before, device, after):
        return before, device, after

    seen = []
    concrete = object()

    def value(label, result):
        seen.append(label)
        return result

    def lookup(content):
        seen.append("lookup")
        assert content is concrete
        return content

    env = {
        "construct": construct,
        "value": value,
        "concrete": concrete,
        "_PS": SimpleNamespace(lookup_global_info=lookup),
    }
    _, rewritten = _rewrite(
        'construct(value("before", 1), value("device", concrete), value("after", 2))', env
    )
    assert _evaluate(rewritten, env) == (1, concrete, 2)
    assert seen == ["before", "device", "lookup", "after"]


def test_unmarked_and_scalar_shorthand_strings_remain_literal():
    @protocol.args_policy({"shape": "expr_str"}, scalar_strings=False)
    def tensor(shape, dtype=None):
        return shape, dtype

    env = {"tensor": tensor}
    _, rewritten = _rewrite('tensor("float32", dtype="literal[0]")', env)
    assert _evaluate(rewritten, env) == ("float32", "literal[0]")
    _, unregistered = _rewrite('ordinary("n + 1")', {})
    assert ast.unparse(unregistered) == "ordinary('n + 1')"


def test_policy_registration_copies_mapping_and_preserves_aliases():
    fields = {"shape": "expr_str", "device": "global_info"}
    decorate = protocol.args_policy(fields)
    fields["shape"] = "global_info"

    def constructor(shape, device):
        return shape, device

    wrapped = decorate(constructor)
    alias = wrapped
    policy = protocol.get_args_policy(alias)
    assert policy is protocol.get_args_policy(constructor)
    assert dict(policy.fields) == {"shape": "expr_str", "device": "global_info"}
    with pytest.raises(TypeError):
        policy.fields["shape"] = "global_info"
    assert protocol.get_args_policy([]) is None
    assert protocol.get_args_policy(None) is None


def test_unknown_policy_and_parameter_are_rejected():
    with pytest.raises(ValueError, match="Unknown argument policies"):
        protocol.args_policy({"shape": "vdevice"})
    with pytest.raises(ValueError, match="Unknown argument policy fields"):
        protocol.args_policy({"missing": "expr_str"})(lambda shape: shape)


def test_eager_annotations_preserve_missing_type_and_concrete_construction():
    @protocol.args_policy({"shape": "expr_str"}, scalar_strings=False)
    def tensor(shape):
        return shape

    assert tensor(("n",)).is_missing()
    assert tensor((TypeVar("n"),)).is_missing()
    assert tensor((16,)) == (16,)
    with IRBuilder(), TypeVarFrame() as symbols:
        n = symbols.resolve("n")
        assert tensor((TypeVar("n"),))[0].same_as(n)
        with pytest.raises(TypeError, match="require concrete symbols"):
            tensor(("n",))


def test_annotation_class_and_shorthand_expression_policy():
    @protocol.args_policy({"shape": "expr_str"}, as_type=True)
    def annotation(shape):
        return shape

    assert isinstance(annotation, type)
    assert annotation | None
    assert annotation(16) == 16
    assert annotation("n").is_missing()

    @protocol.expr_str_args("values", scalar_strings=False)
    def shorthand(values):
        return values

    assert dict(protocol.get_args_policy(shorthand).fields) == {"values": "expr_str"}
    assert protocol.expr_str_policy(shorthand).fields == ("values",)
    env = {"shorthand": shorthand, "X": SimpleNamespace(resolve_type_var=lambda name: 8)}
    _, rewritten = _rewrite('shorthand(("n",))', env)
    assert _evaluate(rewritten, env) == (8,)


def test_quoted_annotation_keeps_signature_symbol_binding():
    @protocol.args_policy({"shape": "expr_str"}, scalar_strings=False)
    def tensor(shape):
        return shape

    env = {"tensor": tensor, "X": SimpleNamespace(resolve_type_var=lambda name: 8)}
    _, rewritten = _rewrite(repr('tensor(("n", n))'), env, annotation=True)
    assert _evaluate(rewritten, env) == (8, 8)
    assert env["n"] == 8


def test_starred_arguments_preserve_valid_python_ast():
    @protocol.args_policy({"device": "global_info"})
    def construct(device=None, *, other=None):
        return device, other

    concrete = object()
    env = {"construct": construct, "values": (concrete,)}
    _, rewritten = _rewrite("construct(*values, other=3)", env)
    assert _evaluate(rewritten, env) == (concrete, 3)


def test_rewriting_nested_constructor_applies_global_policy_once():
    @protocol.args_policy({"device": "global_info"})
    def construct(device):
        return device

    seen = []

    def lookup(content):
        seen.append(content)
        return "resolved " + content

    env = {
        "outer": lambda value: value,
        "construct": construct,
        "_PS": SimpleNamespace(lookup_global_info=lookup),
    }
    _, rewritten = _rewrite('outer(construct("cuda:1"))', env)
    rewritten = rewrite_expression(
        rewritten,
        lambda node: env.get(node.id) if isinstance(node, ast.Name) else None,
        "X",
        "<args-policy>",
    )
    assert _evaluate(rewritten, env) == "resolved cuda:1"
    assert seen == ["cuda:1"]
