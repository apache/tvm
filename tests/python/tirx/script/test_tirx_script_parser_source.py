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
"""Source and span tests for the canonical parser"""

import tvm_ffi
from tvm_ffi import structural_walk

import tvm
import tvm.testing
from tvm.ir import Call, SequentialSpan, TensorLoad, assert_structural_equal
from tvm.script import tirx as T
from tvm.script.parser.inspect_source import Source
from tvm.script.tirx import tile as Tx
from tvm.tirx.stmt import TilePrimitiveCall


def _capture_source(sources):
    """Keep original source coordinates before a definition-site construction."""

    def capture(function):
        sources.append(Source(function))
        return function

    return capture


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
    structural_walk(func.body, nodes.append, order="post")
    matches = [node for node in nodes if predicate(node)]
    assert len(matches) == 1
    return matches[0]


def test_parser_attaches_span_to_direct_call():
    sources = []

    @T.prim_func
    @_capture_source(sources)
    def direct_call():
        T.device_entry()
        barriers = T.alloc_buffer((1,), "uint64", scope="shared")
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

    assert _span_range(call.span) == _span_range(source.to_span(call_ast))


def test_parser_attaches_span_to_nested_tensor_load():
    sources = []

    @T.prim_func
    @_capture_source(sources)
    def nested_load():
        source_buffer = T.alloc_buffer((1,), "int32")
        output = T.alloc_buffer((1,), "int32")
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

    assert _span_range(load.span) == _span_range(source.to_span(load_ast))


def test_parser_retains_inline_call_site_and_definition_spans():
    @T.inline
    def wait_impl(barrier):
        T.cuda.mbarrier_wait(barrier, 0)

    wait_source = Source(wait_impl.__wrapped__)
    wait_call_ast = wait_source.as_ast().body[0].body[0].value
    wait = wait_impl

    sources = []

    @T.prim_func
    @_capture_source(sources)
    def inline_call():
        T.device_entry()
        barriers = T.alloc_buffer((1,), "uint64", scope="shared")
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

    assert isinstance(call.span, SequentialSpan)
    assert [_span_range(span) for span in call.span.spans] == [
        _span_range(caller_source.to_span(caller_call_ast)),
        _span_range(wait_source.to_span(wait_call_ast)),
    ]


def test_parser_attaches_span_to_tile_primitive_call():
    sources = []

    @T.prim_func
    @_capture_source(sources)
    def tile_call():
        A = T.alloc_buffer((16,), "float32")
        Tx.memset(A[0:16], T.float32(0))

    source = sources[0]
    call_ast = source.as_ast().body[0].body[-1].value
    func = tile_call
    call = _find_ir_node(func, lambda node: isinstance(node, TilePrimitiveCall))

    assert _span_range(call.span) == _span_range(source.to_span(call_ast))


def test_parser_spans_do_not_affect_structural_identity():
    source_a = """@T.prim_func\ndef f():\n    T.evaluate(1)\n"""
    source_b = """\n\n@T.prim_func\ndef f():\n    T.evaluate(1)\n"""

    func_a = tvm.script.from_source(source_a, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})
    func_b = tvm.script.from_source(source_b, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})

    assert _span_range(func_a.body.span) == ("<str>", 3, 5, 3, 18)
    assert _span_range(func_b.body.span) == ("<str>", 5, 5, 5, 18)
    assert tvm_ffi.structural_hash(func_a) == tvm_ffi.structural_hash(func_b)
    assert_structural_equal(func_a, func_b)


if __name__ == "__main__":
    tvm.testing.main()
