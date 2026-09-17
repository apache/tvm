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
"""Test shared tensor regions and their dialect boundaries."""

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.ir import PrimType, Range, TensorRegion, Tuple, TupleType, Var


def test_generic_source_and_serialization():
    source = Tuple([Var("value", "float32")])
    ty = TupleType([PrimType("float32")])
    span = tvm.ir.Span(tvm.ir.SourceName("region"), 1, 1, 1, 8)
    region = TensorRegion(source, [Range(2, 6)], ty, span)

    assert region.source.same_as(source)
    assert region.ty.same_as(ty)
    assert region.span.same_as(span)
    assert type(region) is TensorRegion
    assert not hasattr(region, "buffer")
    tvm.ir.assert_structural_equal(region.region, [Range(2, 6)])
    encoded = tvm.ir.save_json(region)
    assert "ir.TensorRegion" in encoded
    assert "tirx.BufferRegion" not in encoded
    restored = tvm.ir.load_json(encoded)
    assert type(restored) is TensorRegion
    tvm.ir.assert_structural_equal(restored, region, map_free_vars=True)
    assert tvm_ffi.structural_hash(restored, map_free_vars=True) == tvm_ffi.structural_hash(
        region, map_free_vars=True
    )


def test_source_definition_pattern_and_free_range_variables():
    n, m = Var("n", "int32"), Var("m", "int32")
    a = tvm.tirx.decl_buffer((n,), "float32", name="a")
    b = tvm.tirx.decl_buffer((m,), "float32", name="b")
    lhs = tvm.tirx.BufferRegion(a, [Range(n)])
    rhs = tvm.tirx.BufferRegion(b, [Range(m)])

    # The source defines its symbolic shape, which is then used by the ranges.
    tvm.ir.assert_structural_equal(lhs, rhs)
    assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)

    i, j = Var("i", "int32"), Var("j", "int32")
    lhs = tvm.tirx.BufferRegion(a, [Range.from_min_extent(i, n)])
    rhs = tvm.tirx.BufferRegion(b, [Range.from_min_extent(j, m)])
    assert not tvm_ffi.structural_equal(lhs, rhs)
    tvm.ir.assert_structural_equal(lhs, rhs, map_free_vars=True)


def test_source_definition_pattern_preserves_shared_identity():
    source, other = Var("source", "float32"), Var("other", "float32")
    replacement = Var("replacement", "float32")
    ty = TupleType([])
    lhs = TensorRegion(source, [Range(4)], ty)
    rhs = TensorRegion(replacement, [Range(4)], ty)
    tvm.ir.assert_structural_equal(lhs, rhs)
    tvm.ir.assert_structural_equal(Tuple([lhs, source]), Tuple([rhs, replacement]))
    assert not tvm_ffi.structural_equal(Tuple([lhs, source]), Tuple([rhs, other]))


def test_tuple_source_definition_pattern_preserves_shared_identity():
    value, replacement, other = [Var(name, "float32") for name in ("value", "replacement", "other")]
    source = Tuple([value, value])
    equivalent_source = Tuple([replacement, replacement])
    distinct_source = Tuple([replacement, other])
    ty = TupleType([])
    lhs = TensorRegion(source, [Range(4)], ty)
    rhs = TensorRegion(equivalent_source, [Range(4)], ty)
    different = TensorRegion(distinct_source, [Range(4)], ty)

    tvm.ir.assert_structural_equal(lhs, rhs)
    assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
    tvm.ir.assert_structural_equal(Tuple([lhs, value]), Tuple([rhs, replacement]))
    assert not tvm_ffi.structural_equal(lhs, different, map_free_vars=True)
    assert not tvm_ffi.structural_equal(Tuple([lhs, value]), Tuple([rhs, other]))


@pytest.mark.parametrize("boundary", ["match_buffer", "sblock_reads", "sblock_writes", "tile_call"])
@pytest.mark.parametrize("invalid_source", ["generic_expr", "wrong_buffer_rank"])
def test_native_buffer_boundaries_reject_invalid_regions(boundary, invalid_source):
    buffer = tvm.tirx.decl_buffer((4,), "float32")
    if invalid_source == "generic_expr":
        region = TensorRegion(Tuple([]), [Range(4)], tvm.tirx.BufferRegionType())
        error_type, error_message = TypeError, r"Cannot treat type `ir\.Tuple` as type `ir\.Var`"
    else:
        # The shared constructor intentionally permits generic source/range pairs;
        # buffer-specific consumers must enforce rank even without the factory.
        region = TensorRegion(buffer, [Range(4), Range(4)], tvm.tirx.BufferRegionType())
        error_type, error_message = tvm.error.InternalError, "must match its buffer rank"

    consumers = {
        "match_buffer": lambda: tvm.tirx.MatchBufferRegion(buffer, region),
        "sblock_reads": lambda: tvm.tirx.SBlock([], [region], [], "read", tvm.tirx.Evaluate(0)),
        "sblock_writes": lambda: tvm.tirx.SBlock([], [], [region], "write", tvm.tirx.Evaluate(0)),
        "tile_call": lambda: tvm.tirx.TilePrimitiveCall(
            region, region, op=tvm.ir.Op.get("tirx.tile.copy")
        ),
    }
    with pytest.raises(error_type, match=error_message):
        consumers[boundary]()


def test_structural_walk_and_map_generic_fields():
    source, replacement = Var("source", "float32"), Var("replacement", "float32")
    n, m = Var("n", "int32"), Var("m", "int32")
    ty, replacement_ty = TupleType([PrimType("float16")]), TupleType([PrimType("float32")])
    region = TensorRegion(source, [Range(n)], ty)
    seen = []
    tvm_ffi.structural_walk(region, lambda node: seen.append(node))
    assert any(node.same_as(source) for node in seen)
    assert any(node.same_as(n) for node in seen)
    assert any(node.same_as(ty) for node in seen)

    unchanged = tvm_ffi.structural_map(region, (Var, lambda node: node))
    assert unchanged.same_as(region)
    remap = {source: replacement, n: m}
    mapped = tvm_ffi.structural_map(
        region,
        [(Var, lambda node: remap.get(node, node)), (TupleType, lambda _: replacement_ty)],
    )
    assert not mapped.same_as(region)
    assert mapped.source.same_as(replacement)
    assert mapped.ty.same_as(replacement_ty)
    assert mapped.region[0].extent.same_as(m)
    assert region.source.same_as(source)
    assert region.ty.same_as(ty)
    assert region.region[0].extent.same_as(n)


def test_buffer_factory_and_subscripts():
    buffer = tvm.tirx.decl_buffer((8, 16), "float32x4")
    region = tvm.tirx.BufferRegion(buffer, [Range(1, 7), Range(2, 14)])
    assert type(region) is TensorRegion
    assert isinstance(region.ty, tvm.tirx.BufferRegionType)
    assert region.source.same_as(buffer)
    narrowed = region[1:3, 2:5]
    assert type(narrowed) is TensorRegion
    assert narrowed.source.same_as(buffer)
    tvm.ir.assert_structural_equal(narrowed.region, [Range(2, 4), Range(4, 7)])
    point = region[1, 2]
    assert isinstance(point, tvm.ir.TensorLoad)
    assert point.source.same_as(buffer)
    assert point.ty == PrimType("float32x4")
    tvm.ir.assert_structural_equal(point.indices, [tvm.tirx.const(2), tvm.tirx.const(4)])
    with pytest.raises(tvm.error.InternalError, match="Buffer rank and region dimension mismatch"):
        tvm.tirx.BufferRegion(buffer, [Range(8)])


def test_buffer_boundaries_reject_generic_source():
    from tvm.tirx.script.builder.ir import match_buffer
    from tvm.tirx.script.builder.tirx import _to_region
    from tvm.tirx.script.parser.parser import slice_buffer_from_region
    from tvm.tirx.script.tile import _require_buffer_arg

    region = TensorRegion(Tuple([]), [Range(4)], TupleType([]))
    for consume in [
        _to_region,
        match_buffer,
        slice_buffer_from_region,
        lambda value: _require_buffer_arg("copy", "dst", value),
    ]:
        with pytest.raises(TypeError, match="BufferVar source"):
            consume(region)


if __name__ == "__main__":
    tvm.testing.main()
