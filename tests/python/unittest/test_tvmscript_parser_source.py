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
"""Unittests for tvm.script.parser.core"""
import pytest
import inspect
import tvm.testing
from tvm.script._parser.core.diagnostics import Source
from tvm.script._parser.core import doc_core as doc
from tvm.script import tir as T


def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
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


if __name__ == "__main__":
    tvm.testing.main()
