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
"""S-TIR node ownership and serialization compatibility."""

import json

import pytest

import tvm
import tvm.testing
from tvm import s_tir, tirx


@pytest.mark.parametrize("legacy", [False, True])
def test_sblock_serialization(legacy):
    source = tirx.decl_buffer((4,), "float32", name="source")
    target = tirx.decl_buffer((4,), "float32", name="target")
    region = tirx.BufferRegion(source, [tvm.ir.Range(0, 4)])
    match = s_tir.MatchBufferRegion(target, region)
    block = s_tir.SBlock([], [region], [region], "copy", tirx.Evaluate(0), match_buffers=[match])
    realize = s_tir.SBlockRealize([], True, block)
    graph = json.loads(tvm.ir.save_json([block, realize, match]))
    type_keys = {node.get("type") for node in graph["nodes"]}
    for name in ("SBlock", "SBlockRealize", "MatchBufferRegion"):
        assert f"s_tir.{name}" in type_keys
        assert f"tirx.{name}" not in type_keys
        assert getattr(s_tir, name).__module__ == "tvm.s_tir.stmt"
        assert not hasattr(tirx, name)
        assert not hasattr(tirx.stmt, name)
    if legacy:
        for node in graph["nodes"]:
            if node.get("type") in {
                "s_tir.SBlock",
                "s_tir.SBlockRealize",
                "s_tir.MatchBufferRegion",
            }:
                node["type"] = node["type"].replace("s_tir.", "tirx.")
    restored_block, restored_realize, restored_match = tvm.ir.load_json(json.dumps(graph))
    assert isinstance(restored_block, s_tir.SBlock)
    assert isinstance(restored_realize, s_tir.SBlockRealize)
    assert isinstance(restored_match, s_tir.MatchBufferRegion)
    assert isinstance(restored_block, tirx.Stmt)
    assert isinstance(restored_realize, tirx.Stmt)
    assert restored_realize.block.same_as(restored_block)
    assert restored_block.match_buffers[0].same_as(restored_match)
    assert restored_block.reads[0].same_as(restored_block.writes[0])
    assert restored_match.source.same_as(restored_block.reads[0])
    tvm.ir.assert_structural_equal(restored_realize, realize, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
