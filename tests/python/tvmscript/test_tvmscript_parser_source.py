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
# ruff: noqa: F401
"""Unittests for tvm.script.parser.core"""

import inspect

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.ir import Call, SequentialSpan, TensorLoad, assert_structural_equal
from tvm.script import tirx as T
from tvm.script.parser.core import doc_core as doc
from tvm.script.parser.core.diagnostics import Source
from tvm.script.tirx import tile as Tx
from tvm.tirx.stmt import TilePrimitiveCall
from tvm.tirx.stmt_functor import post_order_visit


def _tirx_source(func):
    """Leave a function intact while marking its source as TIRx."""
    return func


_tirx_source.dispatch_token = "tirx"


def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j, k in T.grid(128, 128, 128):
        with T.sblock("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def test_source_base():
    source = Source(matmul)
    assert (
        source.source_name == inspect.getsourcefile(matmul)
        and source.start_line is not None
        and source.start_column == 0
        and source.source == inspect.getsource(matmul)
        and source.full_source == inspect.getsource(inspect.getmodule(matmul))
    )


def test_source_ast():
    source = Source(matmul)
    mod = source.as_ast()
    assert isinstance(mod, doc.Module)
    func_def = mod.body[0]
    assert isinstance(func_def, doc.FunctionDef)
    assert func_def.name == "matmul"
    func_args = func_def.args
    assert (
        len(func_args.args) == 3
        and func_args.args[0].arg == "a"
        and func_args.args[1].arg == "b"
        and func_args.args[2].arg == "c"
    )
    func_body = func_def.body
    assert len(func_body) == 4
    func_assigns = func_body[:3]
    assert (
        isinstance(func_assigns[0], doc.Assign)
        and func_assigns[0].targets[0].id == "A"
        and isinstance(func_assigns[1], doc.Assign)
        and func_assigns[1].targets[0].id == "B"
        and isinstance(func_assigns[2], doc.Assign)
        and func_assigns[2].targets[0].id == "C"
    )
    func_for = func_body[3]
    assert (
        len(func_for.target.elts) == 3
        and func_for.target.elts[0].id == "i"
        and func_for.target.elts[1].id == "j"
        and func_for.target.elts[2].id == "k"
    )
    for_body = func_for.body
    assert len(for_body) == 1
    for_block = for_body[0]
    assert isinstance(for_block, doc.With) and len(for_block.body) == 2


def _span_range(span):
    return (
        span.source_name.name,
        span.line,
        span.column,
        span.end_line,
        span.end_column,
    )


def _find_ir_node(func, predicate):
    nodes = []
    post_order_visit(func.body, nodes.append)
    matches = [node for node in nodes if predicate(node)]
    assert len(matches) == 1
    return matches[0]


def test_source_to_span_matches_parser_diagnostic_coordinates():
    source = Source(matmul)
    assign = source.as_ast().body[0].body[0]
    span = source.to_span(assign)
    expected_location = (
        source.start_line + 1,
        5,
        source.start_line + 1,
        38,
    )

    assert source.location(assign) == expected_location
    assert _span_range(span) == (source.source_name, *expected_location)


def test_parser_attaches_span_to_direct_call():
    @_tirx_source
    def direct_call():
        T.device_entry()
        barriers = T.alloc_buffer((1,), "uint64", scope="shared")
        T.cuda.mbarrier_wait(
            T.address_of(barriers[0]),
            0,
        )

    source = Source(direct_call)
    call_ast = source.as_ast().body[0].body[-1].value
    func = T.prim_func(direct_call)
    call = _find_ir_node(
        func,
        lambda node: (
            isinstance(node, Call) and getattr(node.op, "name", None) == "tirx.cuda.mbarrier_wait"
        ),
    )

    assert _span_range(call.span) == _span_range(source.to_span(call_ast))


def test_parser_attaches_span_to_nested_tensor_load():
    @_tirx_source
    def nested_load():
        source_buffer = T.alloc_buffer((1,), "int32")
        output = T.alloc_buffer((1,), "int32")
        output[0] = source_buffer[0] + 1

    source = Source(nested_load)
    load_ast = source.as_ast().body[0].body[-1].value.left
    func = T.prim_func(nested_load)
    load = _find_ir_node(
        func,
        lambda node: (
            isinstance(node, TensorLoad) and getattr(node.source, "name", None) == "source_buffer"
        ),
    )

    assert _span_range(load.span) == _span_range(source.to_span(load_ast))


def test_parser_retains_inline_call_site_and_definition_spans():
    def wait_impl(barrier):
        T.cuda.mbarrier_wait(barrier, 0)

    wait_source = Source(wait_impl)
    wait_call_ast = wait_source.as_ast().body[0].body[0].value
    wait = T.inline(wait_impl)

    @_tirx_source
    def inline_call():
        T.device_entry()
        barriers = T.alloc_buffer((1,), "uint64", scope="shared")
        wait(T.address_of(barriers[0]))

    caller_source = Source(inline_call)
    caller_call_ast = caller_source.as_ast().body[0].body[-1].value
    func = T.prim_func(inline_call)
    call = _find_ir_node(
        func,
        lambda node: (
            isinstance(node, Call) and getattr(node.op, "name", None) == "tirx.cuda.mbarrier_wait"
        ),
    )

    assert isinstance(call.span, SequentialSpan)
    assert [_span_range(span) for span in call.span.spans] == [
        _span_range(caller_source.to_span(caller_call_ast)),
        _span_range(wait_source.to_span(wait_call_ast)),
    ]


def test_parser_attaches_span_to_tile_primitive_call():
    @_tirx_source
    def tile_call():
        A = T.alloc_buffer((16,), "float32")
        Tx.memset(A[0:16], T.float32(0))

    source = Source(tile_call)
    call_ast = source.as_ast().body[0].body[-1].value
    func = T.prim_func(tile_call)
    call = _find_ir_node(func, lambda node: isinstance(node, TilePrimitiveCall))

    assert _span_range(call.span) == _span_range(source.to_span(call_ast))


def test_parser_spans_do_not_affect_structural_identity():
    source_a = """@T.prim_func\ndef f():\n    T.evaluate(1)\n"""
    source_b = """\n\n@T.prim_func\ndef f():\n    T.evaluate(1)\n"""

    func_a = tvm.script.from_source(source_a)
    func_b = tvm.script.from_source(source_b)

    assert _span_range(func_a.body.span) == ("<str>", 3, 5, 3, 18)
    assert _span_range(func_b.body.span) == ("<str>", 5, 5, 5, 18)
    assert tvm_ffi.structural_hash(func_a) == tvm_ffi.structural_hash(func_b)
    assert_structural_equal(func_a, func_b)


def test_nesting_parsing():
    class dummy:
        pass

    for i in range(1):

        @tvm.script.ir_module
        class Module:
            @T.prim_func(s_tir=True)
            def impl(
                A: T.Buffer((12, 196, 64), "float32"),
            ) -> None:
                T.evaluate(0)


if __name__ == "__main__":
    tvm.testing.main()
