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
from tvm.ir.json_compact import upgrade_json


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("legacy_region", [False, True])
def test_sblock_serialization(legacy, legacy_region):
    source = tirx.decl_buffer((4,), "float32", name="source")
    target = tirx.decl_buffer((4,), "float32", name="target")
    span = tvm.ir.Span(tvm.ir.SourceName("region"), 2, 3, 4, 5)
    region = tvm.ir.TensorRegion(source, [tvm.ir.Range(0, 4)], tirx.BufferRegionType(), span)
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
    assert "ir.TensorRegion" in type_keys
    assert "tirx.BufferRegion" not in type_keys
    if legacy_region:
        for node in graph["nodes"]:
            if node.get("type") == "ir.TensorRegion":
                node["type"] = "tirx.BufferRegion"
                node["data"]["buffer"] = node["data"].pop("source")
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
    restored_region = restored_match.source
    assert isinstance(restored_region, tvm.ir.TensorRegion)
    assert isinstance(restored_region.ty, tirx.BufferRegionType)
    assert restored_region.span.source_name.name == "region"
    assert restored_region.span.line == 2
    assert restored_region.span.end_line == 3
    assert restored_region.span.column == 4
    assert restored_region.span.end_column == 5
    tvm.ir.assert_structural_equal(restored_realize, realize, map_free_vars=True)
    canonical = json.loads(tvm.ir.save_json([restored_block, restored_realize, restored_match]))
    assert not any(
        node.get("type")
        in {
            "tirx.BufferRegion",
            "tirx.SBlock",
            "tirx.SBlockRealize",
            "tirx.MatchBufferRegion",
        }
        for node in canonical["nodes"]
    )


def test_untyped_buffer_region_serialization():
    source = tirx.decl_buffer((4,), "float32", name="source")
    region = [tvm.ir.Range(0, 4)]
    graph = json.loads(tvm.ir.save_json([source, region]))
    nodes = graph["nodes"]
    source_index, region_index = nodes[graph["root_index"]]["data"]
    # Before c836e8c942, BufferRegion inherited PrimExprConvertible (an Object),
    # and reflection registered exactly buffer/region, with no type or span.
    # Construct that historical schema directly, independently of TensorRegion
    # serialization, while using current buffer/range schemas for dependencies.
    legacy_index = len(nodes)
    for _ in range(2):
        nodes.append(
            {
                "type": "tirx.BufferRegion",
                "data": {"buffer": source_index, "region": region_index},
            }
        )
    graph["root_index"] = len(nodes)
    nodes.append({"type": "ffi.Array", "data": [legacy_index, legacy_index, legacy_index + 1]})
    legacy_json = json.dumps(graph)
    upgraded = json.loads(upgrade_json(legacy_json))
    assert upgraded["root_index"] == graph["root_index"]
    assert len(upgraded["nodes"]) == len(nodes) + 1
    assert upgraded["nodes"][:legacy_index] == nodes[:legacy_index]
    assert upgraded["nodes"][graph["root_index"]] == nodes[graph["root_index"]]
    for index in (legacy_index, legacy_index + 1):
        assert upgraded["nodes"][index] == {
            "type": "ir.TensorRegion",
            "data": {
                "source": source_index,
                "region": region_index,
                "ty": len(nodes),
                "span": 0,
            },
        }
    first, repeated, second = tvm.ir.load_json(legacy_json)
    assert first.same_as(repeated)
    assert not first.same_as(second)
    assert first.source.same_as(second.source)
    assert first.region.same_as(second.region)
    assert first.ty.same_as(second.ty)
    assert isinstance(first.ty, tirx.BufferRegionType)
    assert first.span is None
    expected = tvm.ir.TensorRegion(source, region, tirx.BufferRegionType())
    tvm.ir.assert_structural_equal(first, expected, map_free_vars=True)


@pytest.mark.parametrize("data", [None, {"region": 0}])
def test_malformed_legacy_buffer_region(data):
    node = {"type": "tirx.BufferRegion"}
    if data is not None:
        node["data"] = data
    with pytest.raises(ValueError, match="requires a buffer field"):
        upgrade_json(json.dumps({"nodes": [{"type": "None"}, node], "root_index": 1}))


@pytest.mark.parametrize("field", ["reads", "writes", "match_source"])
@pytest.mark.parametrize("invalid", ["source", "rank"])
def test_region_requires_buffer_source_and_rank(field, invalid):
    source = (
        tirx.Var("source", "int32") if invalid == "source" else tirx.decl_buffer((4, 4), "float32")
    )
    region = tvm.ir.TensorRegion(source, [tvm.ir.Range(0, 4)], tirx.BufferRegionType())
    message = None if invalid == "source" else "must match its buffer rank"
    with pytest.raises((TypeError, tvm.error.InternalError), match=message):
        if field == "match_source":
            s_tir.MatchBufferRegion(tirx.decl_buffer((4,), "float32"), region)
        else:
            s_tir.SBlock(
                [],
                [region] if field == "reads" else [],
                [region] if field == "writes" else [],
                "invalid",
                tirx.Evaluate(0),
            )


if __name__ == "__main__":
    tvm.testing.main()
