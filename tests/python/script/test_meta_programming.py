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


"""Capture surrounding Python values in annotations, bodies and macros.

These scripts distinguish definition-time annotation scope from body lexical
lookup, and preserve captured identities, side effects and capture lifetimes.
"""

from __future__ import annotations

import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tvm import ir
from tvm.script import ir as I
from tvm.script.parser import entry, protocol_registry

EXTENT = 11
VALUE = 1
MACRO_VALUE = 2


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


def test_unrelated_same_file_caller_does_not_supply_annotation_locals(language):
    # Before: unrelated caller locals shadow both annotation and body global names.
    # Expected builder: X.arg receives global 11; X.record receives global 1.
    M = language.M

    def build():
        @M.function
        def function(output: M.Tensor((EXTENT,), "int32")):
            M.record(VALUE)

        return function

    def caller():
        EXTENT = 99  # noqa: F841
        VALUE = 99  # noqa: F841
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
    # X.arg("x", X.Tensor((definition_extent,))); local_extent = X.bind_(2, name="local_extent")
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
    @I.ir_module
    class Module:
        @M.function
        def first(x: M.Tensor(("n",), "float32")):
            n = M.symbol()  # noqa: F841
            return x

        @M.function
        def second(x: M.Tensor(("n",), "float32")):
            n = M.symbol()  # noqa: F841
            return x

    return Module


def test_dynamic_caller_symbol_does_not_join_function_declarations(language):
    # Before: two function signatures declare "n" while a caller has its own n.
    # Expected builder: each function frame resolves its own n, independent of the caller.
    n = ir.Var("n", "int64")
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


def test_macro_capture_policy_remains_explicit(language, monkeypatch):
    # A changed global must distinguish captured macro scope from caller scope: values 2 then 1.
    M = language.M
    M.macro = protocol_registry.declaration_kind("M.macro", "helper")(entry.make_macro_decorator(M))
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
    M.macro = protocol_registry.declaration_kind("M.macro", "helper")(entry.make_macro_decorator(M))
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
    M.macro = protocol_registry.declaration_kind("M.macro", "helper")(entry.make_macro_decorator(M))
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
