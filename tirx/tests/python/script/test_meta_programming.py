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
"""Shared parser meta programming."""

from __future__ import annotations

import gc
import weakref

# Script-local bindings are observed through the constructed IR.
# ruff: noqa: F841
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tvm import ir
from tvm.script import ir as I
from tvm.script.parser import entry

EXTENT = 11
VALUE = 1
MACRO_VALUE = 2


def test_policies_leave_computed_arguments_and_shorthand_strings_alone(language):
    # Computed arguments must run once without treating their returned strings as symbolic syntax.
    M = language.M
    seen = []
    concrete = object()

    def value(label, result):
        seen.append(label)
        return result

    @M.function
    def main():
        M.Tensor(
            value("shape", (4,)), dtype=value("dtype", "float32"), device=value("device", concrete)
        )
        M.Tensor("float32", placement="literal[0]")

    assert seen == ["shape", "dtype", "device"]
    assert main.body[0][1].args == ((4,), "float32", concrete, "S[0]")
    assert main.body[1][1].args == ("float32", "float32", None, "literal[0]")


def test_nested_policy_and_starred_calls_are_evaluated_once(language):
    # Nested marked calls and starred arguments must preserve their values and evaluation count.
    M = language.M
    seen = []
    mesh = object()
    language.global_infos["mesh[0]"] = mesh
    values = (1, 2)
    n = M.dynamic("n")

    def outer(value):
        seen.append(value)
        return value

    def collect(*values, other):
        return values, other

    @M.function
    def main():
        outer(M.Tensor((n,), device="mesh[0]"))
        collect(*values, other=3)

    assert len(seen) == 1 and seen[0].args[2] is mesh
    dimension = seen[0].args[0][0]
    assert dimension.op == "symbol" and dimension.name == "n"
    assert main.body[1][1] == ((1, 2), 3)


def test_constexpr_is_lazy_and_executes_in_parent_scope(language):
    # Compile-time branches keep Python scope and short-circuit without visiting unselected
    # operands.
    M = language.M
    seen = []

    def choose():
        seen.append("condition")
        return True

    def operand(value):
        seen.append(value)
        return value

    def invalid():
        pytest.fail("an unselected constexpr arm ran")

    @M.function
    def main():
        if M.constexpr(choose()):
            x = 7
        else:
            invalid()
        M.record(x)
        M.record(operand(1) if I.constexpr(operand(True)) else invalid())
        M.record(I.constexpr(operand(0)) and invalid())
        M.record(I.constexpr(operand(4)) or invalid())
        M.record(I.constexpr(operand(2)) and operand(7))
        M.record(I.constexpr(operand(0)) or operand(8))

    assert seen == ["condition", True, 1, 0, 4, 2, 7, 0, 8]
    assert [value for kind, value in main.body] == [7, 1, 0, 4, 7, 8]


def test_ordinary_callable_aliases_update_mutable_targets(language):
    # A local callable must shadow its ambient namesake and preserve ordinary mutable stores.
    M = language.M
    marker, calls = object(), []

    def axis_alias():
        pytest.fail("the shadowed ambient callable ran")

    def ordinary():
        calls.append("ordinary")
        return marker

    @M.function
    def main():
        axis_alias = ordinary
        cell = M.cell()
        cell = axis_alias()
        M.record(cell)

    declaration = next(operands[1] for kind, operands in main.body if kind == "declare")
    stores = [operands for kind, operands in main.body if kind == "set"]
    assert calls == ["ordinary"]
    assert len(stores) == 1 and stores[0][0] is declaration and stores[0][1] is marker
    assert main.body[-1] == ("emit", declaration)


def test_bare_callable_alias_does_not_acquire_constexpr_syntax(language):
    # A bare callable alias must not silently acquire constexpr syntax from the original marker.
    M = language.M
    marker = I.constexpr
    with pytest.raises(TypeError, match="syntax marker"):

        @M.function
        def main():
            if marker(True):
                M.record(1)


def test_ordinary_iterator_binding_calls_the_custom_iterator_once(language):
    # An ordinary iterator alias calls its captured implementation exactly once.
    M = language.M
    calls = []

    def custom_range(extent):
        calls.append(extent)
        return M.grid(2)

    @M.function
    def main():
        iterator = custom_range
        for i in iterator(4):
            M.record(i)

    assert calls == [4]
    variable = main.body[0][1]
    assert variable.op == "loop" and variable.args == (2,) and variable.name == "i"


def test_constexpr_keeps_python_comparison_and_chain_short_circuit(primitive_language):
    # Compile-time comparison chains must short-circuit and retain native identity equality
    # semantics.
    M = primitive_language.M
    seen = []
    x = ir.Var("x", "int32")

    def operand(index, value):
        seen.append(index)
        return value

    def invalid():
        raise AssertionError("constexpr comparison chain must short-circuit")

    @M.function
    def main():
        if M.constexpr(operand(0, 2) < operand(1, 1) < invalid()):
            0
        elif M.constexpr(x == x):
            1

    assert seen == [0, 1]
    assert main.body == [("emit", 1)]


def test_host_lambda_local_does_not_capture_optional_binding(language):
    # A lambda parameter must shadow an outer optional binding during host selection.
    M = language.M

    @M.function
    def main():
        if I.constexpr(False):
            x = 1
        if I.constexpr((lambda x: x)(True)):
            M.record(222)

    assert main.body[-1] == ("emit", 222)


def test_unselected_namespace_call_does_not_require_an_attribute(language):
    # An unselected source call must remain unevaluated even when its captured name resembles a
    # parser helper.
    M = language.M
    lanes = 1

    @M.function
    def main():
        M.record(M.ramp(0, 1, lanes) if I.constexpr(lanes > 1) else 0)

    assert main.body == [("emit", 0)]


def test_attribute_name_does_not_turn_a_closure_binding_into_a_global(language):
    # Before: closure dtype="int32" and descriptor.dtype="int64" share a spelling.
    # Expected builder: X.record receives ("int32", 3), then ("int64", 4).
    M = language.M
    dtype = "int32"
    descriptor = SimpleNamespace(dtype="int64")

    @M.function
    def function():
        M.record((dtype, 3))
        M.record((descriptor.dtype, 4))

    first, second = function.body
    assert first[1][0] == "int32"
    assert first[1][1] == 3
    assert second[1][0] == "int64"
    assert second[1][1] == 4


def test_active_lexical_ancestor_retains_annotation_only_names(language):
    # An annotation must retain an outer-only name while choosing the nearest shadowed name.
    M = language.M

    def outer(width, extent):
        def middle():
            extent = 3

            @M.function
            def function(output: M.Tensor((width, extent), "int32")):
                M.record(VALUE)

            return function

        return middle()

    function = outer(4, 8)
    assert function.params[0].args[0].args[0] == (4, 3)
    assert function.body == [("emit", 1)]


def test_module_parameter_does_not_replace_another_functions_capture(language):
    # Decorator options and later signatures must keep their outer scope despite parameter names.
    M = language.M

    @I.ir_module
    class Module:
        extent = 4

        @M.function(private=extent == 4)
        def first(extent: M.Tensor((extent,))):
            M.record(extent)

        @M.function
        def second(output: M.Tensor((extent,))):
            M.record(output)

    assert Module["first"].params[0].args[0].args[0] == (4,)
    assert Module["second"].params[0].args[0].args[0] == (4,)
    for function in (Module["first"], Module["second"]):
        assert function.body == [("emit", function.params[0])]


def test_body_local_shadows_capture_for_later_annotation(language):
    M = language.M
    extent = 7
    seen = []

    def annotation(shape):
        seen.append(shape)
        return M.Tensor(shape)

    @M.function
    def main(x: M.Tensor((extent,))):
        extent = 3
        value: annotation((extent,)) = x
        M.record(value)

    assert main.params[0].args[0].args[0] == (7,)
    assert seen == [(3,)]
    assert extent == 7
    assert main.body == [("emit", main.params[0])]


def test_unrelated_same_file_caller_does_not_supply_annotation_locals(language):
    # Before: unrelated caller locals shadow both annotation and body global names.
    # Expected builder: X.arg_ receives global 11; X.record receives global 1.
    M = language.M

    def build():
        @M.function
        def function(output: M.Tensor((EXTENT,), "int32")):
            M.record(VALUE)

        return function

    def caller():
        EXTENT = 99
        VALUE = 99
        return build()

    function = caller()
    assert [int(value) for value in function.params[0].args[0].args[0]] == [11]
    assert function.body[0][1] == 1


def test_enclosing_class_locals_are_not_nested_annotation_scope(language):
    # Before: outer class EXTENT=99 surrounds a nested class definition.
    # Expected builder: nested annotation and body read global 11.
    M = language.M

    class Outer:
        EXTENT = 99

        class Inner:
            @M.function
            def function(output: M.Tensor((EXTENT,), "int32")):
                M.record(EXTENT)

    function = Outer.Inner.function
    assert [int(value) for value in function.params[0].args[0].args[0]] == [11]
    assert function.body[0][1] == 11


def test_unrelated_intervening_call_ends_annotation_capture(language):
    # Before: an unrelated invoke(callback) frame interrupts lexical ancestor capture.
    # Expected builder: raise NameError naming unavailable extent.
    M = language.M

    def invoke(callback):
        return callback()

    def outer(extent):
        def middle():
            @M.function
            def function(output: M.Tensor((extent,), "int32")):
                M.record(VALUE)

            return function

        return invoke(middle)

    with pytest.raises(NameError, match="extent"):
        outer(8)


def test_local_annotation_keeps_rhs_before_constructor_evaluation(language):
    # The RHS must run once before its annotation constructor, preserving the returned value.
    M = language.M
    events = []

    def annotation():
        events.append("annotation")
        return M.Tensor((2,), "float32")

    def value(x):
        events.append("rhs")
        return M.value(x, x)

    @M.function
    def function(x: M.Tensor((2,), "float32")):
        y: annotation() = value(x)
        return y

    assert events == ["rhs", "annotation"]
    kind, result = function.body[0]
    assert kind == "return" and result.op == "value"
    assert result.args == (function.params[0], function.params[0])


def test_local_annotation_scope_does_not_overwrite_ordinary_body_global(language):
    # A local annotation reads the class extent and captured dtype; the body still reads global 11.
    M = language.M
    dtype = "float32"
    annotations = []

    def annotation(shape, dtype):
        annotations.append((shape, dtype))
        return M.Tensor(shape, dtype)

    @I.ir_module
    class Module:
        EXTENT = 3

        @M.function
        def main(x: M.Tensor((3,), "float32")):
            M.record(EXTENT)
            y: annotation((EXTENT,), dtype) = M.value(x, x)
            return y

    function = Module["main"]
    assert annotations == [((3,), "float32")]
    assert function.body[0] == ("emit", 11)
    kind, result = function.body[1]
    assert kind == "return" and result.op == "value"
    assert result.args == (function.params[0], function.params[0])


def test_local_annotation_preserves_lambda_and_comprehension_bindings(language):
    # Lambda/comprehension locals in an annotation must not leak into later body captures.
    M = language.M
    dtype = "int64"
    size = 9
    sizes = (2,)
    annotations = []

    def annotate(shape, dtype):
        annotations.append((shape, dtype))
        return M.Tensor(shape, dtype)

    @M.function
    def function(x: M.Tensor((2,), "float32")):
        y: (lambda dtype: annotate(tuple(size for size in sizes), dtype))("float32") = M.value(x, x)
        M.record((dtype, size))
        return y

    assert annotations == [((2,), "float32")]
    assert function.body[0] == ("emit", ("int64", 9))
    kind, result = function.body[1]
    assert kind == "return" and result.op == "value"
    assert result.args == (function.params[0], function.params[0])


def test_class_annotation_scope_keeps_distinct_method_closure(language):
    # Before: an enclosing extent=7 and class extent=3 share a method spelling.
    # Expected builder program: the annotation reads class 3, the body closes over 7.
    M = language.M
    extent = 7

    @I.ir_module
    class Module:
        extent = 3

        @M.function
        def main(value: M.Tensor((extent,))):
            M.record(extent)

    function = Module["main"]
    assert function.params[0].args[0].args[0] == (3,)
    assert function.body == [("emit", 7)]


def test_actual_decorator_preserves_annotation_definition_and_body_scopes(language):
    # A decorated function uses the enclosing extent in its annotation and a body local.
    # Expected builder program:
    # X.arg_("x", X.Tensor((definition_extent,))); local_extent = X.bind_(2, name="local_extent")
    M = language.M
    calls = []

    def ordinary_decorator(function):
        calls.append("decorate")
        return function

    M.ordinary = ordinary_decorator

    def outer(extent):
        @M.function
        def main(x: M.Tensor((extent,))):
            local_extent = 2

            @M.ordinary
            def helper():
                calls.append("call")
                return local_extent

            M.record(helper())

        return main

    result = outer(7)
    assert result.params[0].args[0].args[0] == (7,)
    assert result.body[0][1] == 2
    assert calls == ["decorate", "call"]


def test_pyfunc_collection_preserves_original_callable_and_closure(language):
    # Before: I.pyfunc marks an ordinary closure in a class.
    # Expected builder program: attach the original callable to __pyfuncs__; no IR declaration.
    factor = 7
    calls = []

    class Source:
        @I.pyfunc
        def multiply(value):
            calls.append(value)
            return value * factor

    original = Source.multiply
    result = I.ir_module(Source)
    assert result.__pyfuncs__ == {"multiply": original}
    assert result == {}
    assert calls == []
    assert result.__pyfuncs__["multiply"](3) == 21
    assert calls == [3]


def test_body_preserves_capture_and_parameter_identity(language):
    # Before: the body reads an enclosing token and its original runtime parameter.
    # Expected builder program: preserve both identities under their source names,
    # while preserving their original lexical lookup.
    token = object()
    M = language.M

    @M.function
    def main(x: M.Tensor((4,))):
        M.record(token)
        M.record(x)

    assert main.body[0][1] is token
    assert main.body[1][1] is main.params[0]


def _build_symbolic_functions(M):
    first_n = M.dynamic("n")
    second_n = M.dynamic("n")

    @I.ir_module
    class Module:
        @M.function
        def first(x: M.Tensor((first_n,), "float32")):
            return x

        @M.function
        def second(x: M.Tensor((second_n,), "float32")):
            return x

    return Module


def test_dynamic_caller_symbol_does_not_join_function_declarations(language):
    # Two independently constructed symbols remain distinct from an unrelated caller capture.
    n = I.dynamic("n")
    module = _build_symbolic_functions(language.M)
    first = module["first"].params[0].args[0].args[0][0]
    second = module["second"].params[0].args[0].args[0][0]
    assert first is not second
    assert first is not n
    assert second is not n


def test_nonlocal_declaration_preserves_captured_values(language):
    # A nonlocal declaration must preserve annotation/body captures and the returned parameter.
    M = language.M
    dtype = "float32"
    value = 3

    @M.function
    def main(x: M.Tensor((2,), dtype)):
        nonlocal dtype, value
        M.record(value + 1)
        return x

    assert main.params[0].args[0].args[:2] == ((2,), "float32")
    assert main.body[0] == ("emit", 4)
    assert main.body[1][0] == "return" and main.body[1][1] is main.params[0]
    assert dtype == "float32" and value == 3


@pytest.mark.parametrize("hygienic", [True, False])
def test_macro_local_annotation_captures_definition_and_argument_names(
    language, hygienic, monkeypatch
):
    M = language.M
    M.macro = entry.make_macro_decorator(M, namespace_path="M.macro")
    MACRO_VALUE = 7
    observed = []

    def annotation(shape):
        observed.append(shape)
        return M.Tensor(shape, "float32")

    monkeypatch.setitem(globals(), "annotation", annotation)

    @M.macro(hygienic=hygienic)
    def typed_local(size):
        value: annotation((MACRO_VALUE, size)) = M.value(1, 2)
        M.record(value)

    monkeypatch.setitem(globals(), "MACRO_VALUE", 9)

    @M.function
    def function():
        typed_local(3)

    assert observed == [(7 if hygienic else 9, 3)]
    assert function.body[0][0] == "emit"
    assert function.body[0][1].op == "value"


def test_macro_capture_policy_remains_explicit(language, monkeypatch):
    # A changed global must distinguish captured macro scope from caller scope: values 2 then 1.
    M = language.M
    M.macro = entry.make_macro_decorator(M, namespace_path="M.macro")
    monkeypatch.setitem(globals(), "MACRO_VALUE", 2)

    @M.macro
    def captured():
        return MACRO_VALUE

    @M.macro(hygienic=False)
    def caller_scoped():
        return MACRO_VALUE

    monkeypatch.setitem(globals(), "MACRO_VALUE", 1)

    @M.function
    def function():
        M.record(captured())
        M.record(caller_scoped())

    assert function.body == [("emit", 2), ("emit", 1)]


def test_macro_dynamic_global_lookup_uses_temporary_invocation_namespace(language, monkeypatch):
    # Dynamic global lookup inside a macro must see names absent from its static bytecode reads.
    M = language.M
    M.macro = entry.make_macro_decorator(M, namespace_path="M.macro")
    monkeypatch.setitem(globals(), "_macro_dynamic_value", 7)

    @M.macro
    def helper():
        return globals()["_macro_dynamic_value"]

    @M.function
    def function():
        M.record(helper())

    assert function.body == [("emit", 7)]


class Payload:
    pass


@contextmanager
def without_cyclic_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def test_eager_entry_releases_unused_scope_but_keeps_annotation_value(language):
    # Eager annotation capture must keep width without retaining the unrelated enclosing payload.
    M = language.M

    def make():
        payload = Payload()
        reference = weakref.ref(payload)
        width = 7

        @M.function
        def function(value: M.Tensor((width,))):
            M.record(1)

        return function, reference

    with without_cyclic_gc():
        function, reference = make()
        assert reference() is None
        assert function.params[0].args[0].args[0] == (7,)
        assert function.body == [("emit", 1)]


def test_live_macro_does_not_snapshot_unrelated_module_globals(language, monkeypatch):
    # A live macro must retain its closure value without snapshotting unrelated module globals.
    M = language.M
    M.macro = entry.make_macro_decorator(M, namespace_path="M.macro")
    monkeypatch.setitem(globals(), "_unrelated_macro_payload", Payload())
    reference = weakref.ref(globals()["_unrelated_macro_payload"])
    offset = 3

    with without_cyclic_gc():

        @M.macro
        def helper(value):
            return value + offset

        del globals()["_unrelated_macro_payload"]
        assert reference() is None

        @M.function
        def function():
            M.record(helper(2))

        assert function.body == [("emit", 5)]


def test_constexpr_uses_registered_namespace_in_source(language):
    # An ordinary attribute named constexpr must not acquire the registered marker's binding rule.
    M = language.M

    class Ordinary:
        constexpr = M.Tensor((1,), "int32")

    @M.function
    def ordinary(value: Ordinary.constexpr):
        return value

    assert len(ordinary.params) == 1
    assert ordinary.params[0].args[0].args[:2] == ((1,), "int32")
    assert ordinary.body[0][1] is ordinary.params[0]

    with pytest.raises(TypeError, match="requires a specialization binding"):

        @M.function
        def specialized(value: I.constexpr):
            M.record(value)


def test_definition_defaults_preserve_names_and_nested_body_globals(language, monkeypatch):
    M = language.M
    parse = entry.parse
    recompose = entry._recompose_builder
    captured = []

    def with_definition_scope(source, **kwargs):
        kwargs["definition_scope"]["EXTENT"] = 128
        return parse(source, **kwargs)

    def observe(*args, **kwargs):
        builder = recompose(*args, **kwargs)
        captured.append((builder.__kwdefaults__["EXTENT"], builder.__globals__["EXTENT"]))
        assert not any(name.startswith("_definition_scope") for name in builder.__globals__)
        return builder

    monkeypatch.setattr(entry, "parse", with_definition_scope)
    monkeypatch.setattr(entry, "_recompose_builder", observe)

    @M.function
    def function(x: M.Tensor((EXTENT,))):
        def helper():
            return EXTENT

        M.record(helper())
        M.record((lambda: EXTENT)())
        M.record(tuple(EXTENT for _ in range(1)))

    assert captured == [(128, 11)]
    assert function.params[0].args[0].args[0] == (128,)
    assert function.body == [("emit", 11), ("emit", 11), ("emit", (11,))]


def test_class_captures_follow_each_source_declaration(language):
    M = language.M

    @I.ir_module
    class Module:
        extent = 3

        @M.function
        def first(x: M.Tensor((extent,))):
            M.record(x)

        extent = 5

        @M.function
        def second(x: M.Tensor((extent,))):
            M.record(x)

    assert Module["first"].params[0].args[0].args[0] == (3,)
    assert Module["second"].params[0].args[0].args[0] == (5,)


def test_namespace_alias_preserves_evaluated_root_kwargs(language):
    Alias = language.M
    events = []

    def private():
        events.append("private")
        return True

    @Alias.function(private=private())
    def function(x: Alias.Tensor((2,))):
        Alias.record(x)

    assert events == ["private"]
    assert function.body == [("emit", function.params[0])]


def test_empty_root_kwargs_do_not_reexecute_decorator_arguments(language):
    M = language.M
    calls = []

    def options():
        calls.append("options")
        return {}

    @M.function(**options())
    def function():
        M.record(1)

    assert calls == ["options"]
    assert function.body == [("emit", 1)]


def test_construction_rejects_later_application_even_with_decorator_source(language):
    M = language.M
    originals = []

    def retain(function):
        originals.append(function)
        return function

    @M.function
    @retain
    def function():
        M.record(1)

    assert function.body == [("emit", 1)]
    with pytest.raises(SyntaxError, match="definition site"):
        M.function(originals[0])
    with pytest.raises(SyntaxError, match="definition site"):
        M.function(private=True)(originals[0])


@pytest.mark.parametrize("configured", [False, True])
def test_callable_decorator_alias_has_explicit_restriction(language, configured):
    decorator = language.M.function(private=True) if configured else language.M.function
    with pytest.raises(SyntaxError, match="bare and preconfigured decorator aliases"):

        @decorator
        def function():
            pass
