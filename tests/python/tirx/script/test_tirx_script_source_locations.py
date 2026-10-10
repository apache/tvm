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

"""TIRx script source locations."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import tvm_ffi
from tvm_ffi import structural_walk

import tvm
import tvm.testing
from tvm import ir, tirx
from tvm.ir import Call, CallSiteLoc, TensorLoad, assert_structural_equal, prim
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.ir_builder.base import AlreadyEmitted
from tvm.script.parser.inspect_source import Source


def test_parser_attaches_loc_to_direct_call():
    sources = []

    @T.function
    @_capture_source(sources)
    def direct_call():
        T.device_entry()
        barriers = T.alloc_tensor((1,), "uint64", scope="shared")
        T.cuda.mbarrier_wait(
            T.address_of(barriers[0]),
            0,
        )

    source = sources[0]
    call_ast = source.as_ast().body[0].body[-1].value
    func = direct_call
    call = _find_ir_node(
        func,
        lambda node: (
            isinstance(node, Call) and getattr(node.op, "name", None) == "tirx.cuda.mbarrier_wait"
        ),
    )

    assert _loc_range(call.loc) == _loc_range(source.to_loc(call_ast))


def _capture_source(sources):
    """Keep original source coordinates before a definition-site construction."""

    def capture(function):
        sources.append(Source(function))
        return function

    return capture


def _loc_range(loc):
    return (
        loc.source_name.name,
        loc.start_line,
        loc.start_column,
        loc.end_line,
        loc.end_column,
    )


def _find_ir_node(func, predicate):
    nodes = []
    structural_walk(func.body, nodes.append, order="post")
    matches = [node for node in nodes if predicate(node)]
    assert len(matches) == 1
    return matches[0]


def test_parser_attaches_loc_to_nested_tensor_load():
    sources = []

    @T.function
    @_capture_source(sources)
    def nested_load():
        source_buffer = T.alloc_tensor((1,), "int32")
        output = T.alloc_tensor((1,), "int32")
        output[0] = source_buffer[0] + 1

    source = sources[0]
    load_ast = source.as_ast().body[0].body[-1].value.left
    func = nested_load
    load = _find_ir_node(
        func,
        lambda node: (
            isinstance(node, TensorLoad) and getattr(node.source, "name", None) == "source_buffer"
        ),
    )

    assert _loc_range(load.loc) == _loc_range(source.to_loc(load_ast))


def test_parser_retains_inline_call_site_and_definition_locs():
    @T.inline
    def wait_impl(barrier):
        T.cuda.mbarrier_wait(barrier, 0)

    wait_source = Source(wait_impl.__wrapped__)
    wait_call_ast = wait_source.as_ast().body[0].body[0].value
    wait = wait_impl

    sources = []

    @T.function
    @_capture_source(sources)
    def inline_call():
        T.device_entry()
        barriers = T.alloc_tensor((1,), "uint64", scope="shared")
        wait(T.address_of(barriers[0]))

    caller_source = sources[0]
    caller_call_ast = caller_source.as_ast().body[0].body[-1].value
    func = inline_call
    call = _find_ir_node(
        func,
        lambda node: (
            isinstance(node, Call) and getattr(node.op, "name", None) == "tirx.cuda.mbarrier_wait"
        ),
    )

    assert isinstance(call.loc, CallSiteLoc)
    assert [_loc_range(loc) for loc in (call.loc.caller, call.loc.callee)] == [
        _loc_range(caller_source.to_loc(caller_call_ast)),
        _loc_range(wait_source.to_loc(wait_call_ast)),
    ]


def test_parser_attaches_loc_to_tile_primitive_call():
    sources = []

    @T.function
    @_capture_source(sources)
    def tile_call():
        A = T.alloc_tensor((16,), "float32")
        T.cuda.tile.mov(A[0:16], T.float32(0))

    source = sources[0]
    call_ast = source.as_ast().body[0].body[-1].value
    func = tile_call
    call = _find_ir_node(
        func,
        lambda node: isinstance(node, tvm.ir.Call)
        and isinstance(node.op, tvm.ir.Op)
        and node.op.name == "tirx.cuda.tile.mov",
    )

    assert _loc_range(call.loc) == _loc_range(source.to_loc(call_ast))


def test_parser_locs_do_not_affect_structural_identity():
    source_a = """@T.function\ndef f():\n    T.evaluate(1)\n"""
    source_b = """\n\n@T.function\ndef f():\n    T.evaluate(1)\n"""

    func_a = tvm.script.from_source(source_a, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})
    func_b = tvm.script.from_source(source_b, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})

    assert _loc_range(func_a.body[0].loc) == ("<str>", 3, 5, 3, 18)
    assert _loc_range(func_b.body[0].loc) == ("<str>", 5, 5, 5, 18)
    assert tvm_ffi.structural_hash(func_a) == tvm_ffi.structural_hash(func_b)
    assert_structural_equal(func_a, func_b)


def test_statement_receipts_keep_emitted_nodes_and_locs():
    # Direct statement hooks return the stored nodes, with no extra emit or duplicate statement.
    location = ir.SourceLoc(ir.SourceName("direct_builder.py"), 5, 3, 5, 17)
    with I.IRBuilder():
        with T.function_(private=True) as frame:
            T.func_name_("receipts")
            output = T.arg_("output", T.Tensor((1,), "int32"))
            stored = T.setitem_(value=3, target=output, key=0, loc=location)
            holder = SimpleNamespace(value=output)
            updated = T.setattr_(holder, "value", 4, loc=location)
            with T.while_(T.bool(True)):
                continued = T.continue_(loc=location)
                broken = T.break_(loc=location)
            returned = T.return_(T.int32(7), loc=location)

    body = frame.function.body.seq
    assert len(body) == 4 and len(body[2].body.seq) == 2
    receipts = [stored, updated, continued, broken, returned]
    statements = [body[0], body[1], *body[2].body.seq, body[3]]
    for receipt, statement in zip(receipts, statements):
        assert isinstance(receipt, AlreadyEmitted)
        assert receipt.value.same_as(statement)
        assert statement.loc.same_as(location)


def test_native_bind_keeps_returned_and_stored_variable_identity():
    # Aliases of an explicitly produced native variable must not rename, reloc or bind it again.
    from tvm import ir

    loc = ir.SourceLoc(ir.SourceName("producer.py"), 7, 2, 7, 19)
    produced = ir.Var("producer_name", "int32", loc)
    calls, observed = [], []

    def argument(value):
        calls.append(value)
        return value

    def observe(value):
        observed.append(value)
        assert value.same_as(produced) and value.loc.same_as(loc)
        assert value.name == "producer_name"

    @T.function
    def main(x: T.int32):
        renamed = T.bind(argument(x), var=produced)
        alias = renamed
        called = argument(alias)
        observe(renamed)
        observe(alias)
        observe(called)
        T.evaluate(called)

    binding, use = main.body.seq
    assert isinstance(binding, tvm.ir.Bind) and isinstance(use, tvm.ir.Evaluate)
    assert binding.var.same_as(produced) and use.value.same_as(produced)
    assert len(observed) == 3 and len(calls) == 2
    assert calls[0].same_as(main.params[0]) and calls[1].same_as(produced)


def test_native_view_keeps_producer_identity_name_and_loc(monkeypatch):
    # A native view must preserve its producer name/loc and declare its storage exactly once.
    from functools import wraps

    from tvm import ir
    from tvm.script.ir_builder import base

    original = tvm.tirx.TensorType.view
    seen, produced, observed = [], [], []
    loc = ir.SourceLoc(ir.SourceName("producer.py"), 7, 2, 7, 19)

    @wraps(original)
    def view(ty, buffer, *args):
        seen.append("view")
        value = base.at_(loc, original(ty, buffer, *args))
        produced.append((value, value.name, value.loc))
        return value

    def mark():
        seen.append("argument")
        return 16

    def observe(value):
        observed.append((value, value.loc))

    monkeypatch.setattr(tvm.tirx.TensorType, "view", view)

    @T.function
    def main(A: T.Tensor((4, 4), "float32")):
        renamed = A.view(mark())
        alias = renamed
        observe(renamed)
        observe(alias)
        alias[0] = 0
        A.view(mark())

    captured = main.params[0]
    assert seen == ["argument", "view", "argument", "view"]
    value, name, produced_loc = produced[0]
    assert len(produced) == len(observed) == 2
    assert name != "renamed" and value.name == name
    assert all(
        item.same_as(value) and location.same_as(produced_loc) for item, location in observed
    )
    nodes = list(main.body.seq)
    assert len(nodes) == 3 and isinstance(nodes[0], tvm.ir.Bind)
    assert isinstance(nodes[1], tvm.ir.TensorStore) and isinstance(nodes[2], tvm.ir.Bind)
    assert nodes[0].var.same_as(value) and nodes[1].dest.same_as(value)
    assert nodes[2].var.same_as(produced[1][0])
    ir.assert_structural_equal(nodes[0].value.args[0], captured.data)
    ir.assert_structural_equal(nodes[2].value.args[0], captured.data)


def test_native_binding_preserves_metadata_but_binds_buffer_expressions():
    # Inert metadata must retain resource identity; a non-Var buffer expression still needs a
    # located binding.
    from tvm import ir
    from tvm.script.ir_builder import base

    producer_loc = ir.SourceLoc(ir.SourceName("producer.py"), 7, 2, 7, 19)
    layout = T.TileLayout(T.S[4])

    @T.meta_class
    class Holder:
        def __init__(self, resource):
            self.resource = resource

    values = []

    def initialize(buffer):
        base.at_(producer_loc, buffer)
        values.extend([layout, Holder(buffer), ir.TupleGetItem(ir.Tuple([buffer]), 0)])

    calls, observed, relocs = [], [], []

    def make(index):
        relocs.append(values[1].resource.loc)
        calls.append(index)
        return values[index]

    def observe(*items):
        observed.extend(items)

    @T.function
    def main(A: T.Tensor((4,), "float32")):
        initialize(A)
        renamed_layout = make(0)
        renamed_holder = make(1)
        bound = make(2)
        observe(renamed_layout, renamed_holder)
        make(0)
        make(1)

    buffer = main.params[0]
    holder, projection = values[1:]
    assert calls == [0, 1, 2, 0, 1]
    assert observed[0].same_as(layout)
    assert observed[1] is holder and holder.resource.same_as(buffer)
    assert all(buffer.loc.same_as(loc) for loc in relocs)
    binding = main.body[0]
    assert isinstance(binding, tvm.ir.Bind) and binding.value.same_as(projection)
    assert binding.var.name == "bound"
    assert tirx.is_tensor_var(binding.var)
    assert not isinstance(projection, ir.Var)
    line = _line_of(
        test_native_binding_preserves_metadata_but_binds_buffer_expressions, "bound = make(2)"
    )
    assert (
        binding.var.loc.start_line,
        binding.var.loc.start_column,
        binding.var.loc.end_column,
    ) == (
        line,
        9,
        14,
    )
    assert (projection.loc.start_line, projection.loc.start_column, projection.loc.end_column) == (
        line,
        17,
        24,
    )
    with pytest.raises(TypeError):

        @T.function
        def invalid():
            object()


def _line_of(function, statement):
    lines, first = inspect.getsourcelines(function)
    return first + next(index for index, line in enumerate(lines) if line.strip() == statement)


def test_non_call_expression_reads_keep_their_source_range():
    # Bare name, property, item and literal emissions must retain their written
    # ranges in actual IR, with each host read evaluated once.
    value = prim.IntImm("int32", 1)
    attribute = prim.IntImm("int32", 2)
    item = prim.IntImm("int32", 3)
    reads = []

    class Holder:
        @property
        def value(self):
            reads.append("attribute")
            return attribute

        def __getitem__(self, index):
            reads.append(index)
            return item

    holder = Holder()

    @T.function
    def main():
        value
        holder.value
        holder[0]
        7

    nodes = list(main.body.seq)
    assert [int(node.value) for node in nodes] == [1, 2, 3, 7]
    assert nodes[0].value.same_as(value)
    assert nodes[1].value.same_as(attribute)
    assert nodes[2].value.same_as(item)
    assert reads == ["attribute", 0]
    for node, expression in zip(nodes, ("value", "holder.value", "holder[0]", "7")):
        expected = _position(
            test_non_call_expression_reads_keep_their_source_range, expression, expression
        )
        assert _loc_position(node.loc) == expected
        assert node.loc.source_name.name == __file__


def _position(test, statement, expression):
    lines, start = inspect.getsourcelines(test)
    index, line = next((i, line) for i, line in enumerate(lines) if line.strip() == statement)
    column = line.index(expression) + 1
    return start + index, column, start + index, column + len(expression)


def _loc_position(loc):
    return loc.start_line, loc.start_column, loc.end_line, loc.end_column
