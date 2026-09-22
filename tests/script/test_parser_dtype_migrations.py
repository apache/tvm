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
"""Scoped dtype migrations preserve symbols in the actual migrated sources."""

import ast
import textwrap
from pathlib import Path

import pytest
import tvm_ffi

import tvm
from tvm import ir, tirx
from tvm.script import parser

_ROOT = Path(__file__).resolve().parents[2]
_PAGE_KERNELS = "python/tvm/relax/frontend/nn/llm/_page_kernels.py"


def _source(path, name):
    source = (_ROOT / path).read_text()
    node = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name == name
    )
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
    return textwrap.dedent("\n".join(source.splitlines()[start - 1 : node.end_lineno]))


def _nodes(function, kind):
    nodes = []

    def visit(node):
        if isinstance(node, list | tuple | tvm_ffi.container.Array):
            for item in node:
                visit(item)
        elif isinstance(node, tvm.runtime.Object):
            if isinstance(node, kind):
                nodes.append(node)
            for field in (
                "body",
                "block",
                "seq",
                "then_case",
                "else_case",
                "value",
                "args",
                "indices",
                "a",
                "b",
            ):
                child = getattr(node, field, None)
                if child is not None:
                    visit(child)

    visit(function.body)
    return nodes


@pytest.mark.parametrize("mla", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_page_kernel_source_symbols(mla, debug):
    name = "tir_kv_cache_debug_get_kv" if debug else "tir_kv_cache_transpose_append"
    if mla:
        name += "_mla"
    function = parser.parse(
        _source(_PAGE_KERNELS, name),
        extra_vars={
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "d_qk": 8,
            "dtype": "float16",
            **({} if debug else {"page_size": 16}),
        },
    )
    buffers = [param for param in function.params if tirx.is_buffer_var(param)]
    pages = buffers[0]
    if debug:
        position_map, data = buffers[1:3]
        tokens = position_map.shape[0]
        assert data.shape[1].same_as(tokens)
        if not mla:
            assert buffers[3].shape[1].same_as(tokens)
        page_size = pages.shape[1 if mla else 3]
        assert page_size.ty.dtype == "int64"
        assert page_size.name == "page_size"
        divisions = _nodes(function, tirx.FloorDiv) + _nodes(function, tirx.FloorMod)
        assert divisions
        assert all(node.b.same_as(page_size) for node in divisions)
    else:
        data, position_map = buffers[1], buffers[-1]
        tokens = data.shape[0]
        assert position_map.shape[0].same_as(tokens)
        if not mla:
            assert buffers[2].shape[0].same_as(tokens)
    assert tokens.ty.dtype == "int64"
    assert tokens.name == ("seqlen" if debug else "ntoken")
    loops = _nodes(function, tirx.For)
    assert any(loop.extent.same_as(tokens) for loop in loops)


def test_extern_source_offsets_are_shared_with_packed_arguments():
    function = parser.parse(_source("tests/python/te/test_te_create_primfunc.py", "tir_extern"))
    arrays = [
        call
        for call in _nodes(function, ir.Call)
        if getattr(call.op, "name", "") == "tirx.tvm_stack_make_array"
    ]
    assert len(arrays) == 3
    offsets = [param.elem_offset for param in function.params]
    for index, (offset, array) in enumerate(zip(offsets, arrays), 1):
        assert offset.ty.dtype == "int32"
        assert offset.name == f"off{index}"
        assert array.args[-1].same_as(offset)
    assert all(
        not left.same_as(right) for i, left in enumerate(offsets) for right in offsets[i + 1 :]
    )


def test_illegal_extent_source_keeps_vectorization_diagnostic():
    module = parser.parse(
        _source("tests/python/tirx-transform/test_tir_transform_vectorize.py", "Mod")
    )
    loops = _nodes(module["main"], tirx.For)
    assert len(loops) == 1
    assert loops[0].extent.ty.dtype == "int32"
    assert loops[0].extent.name == "n"
    with pytest.raises(tvm.error.InternalError, match="Failed to vectorize loop with extent n"):
        tirx.transform.VectorizeLoop()(module)
