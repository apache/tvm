/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Tiling operations and helpers for TileLayout.
 */
#include "tile_internal.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

std::pair<TileLayout, std::vector<int64_t>> Group(TileLayout layout,
                                                  const ffi::Array<PrimExpr>& shape) {
  sym::Analyzer analyzer;
  size_t shape_idx = 0;
  PrimExpr prod = 1;

  std::vector<Iter> new_shard;
  std::vector<int64_t> seps{0};

  for (size_t i = 0; i < layout->shard.size(); ++i) {
    auto extent_i = layout->shard[i]->extent;
    auto stride_i = layout->shard[i]->stride;
    prod *= extent_i;
    while (shape_idx < shape.size() &&
           analyzer->CanProveEqual(floormod(prod, shape[shape_idx]), 0)) {
      // Simplify ``c``, ``floordiv(extent_i, c)`` and ``stride_i * c`` —
      // without this, splitting an iter whose extent contains a symbolic
      // dim that algebraically cancels (e.g. ``floordiv(batch_size,
      // batch_size) == 1``) leaves dead ``a // a`` factors in the new
      // iter's stride that ``int(stride)`` can't unwrap downstream.
      PrimExpr c = analyzer->Simplify(floordiv(prod, shape[shape_idx]));
      TVM_FFI_ICHECK(analyzer->CanProveEqual(floormod(extent_i, c), 0))
          << "layout " << layout << " can not be grouped by shape " << shape;
      new_shard.push_back(Iter(analyzer->Simplify(floordiv(extent_i, c)),
                               analyzer->Simplify(stride_i * c), layout->shard[i]->axis));
      extent_i = c;
      prod = c;
      shape_idx++;
      seps.push_back(new_shard.size());
    }
    extent_i = analyzer->Simplify(extent_i);
    if (!IsOne(extent_i)) {
      TVM_FFI_ICHECK(shape_idx < shape.size())
          << "layout " << layout << " can not be grouped by shape " << shape;
      new_shard.push_back(Iter(extent_i, analyzer->Simplify(stride_i), layout->shard[i]->axis));
    }
  }

  TVM_FFI_ICHECK(shape_idx == shape.size())
      << "layout " << layout << " can not be grouped by shape " << shape;

  auto* n = layout.CopyOnWrite();
  n->shard = new_shard;
  return {ffi::GetRef<TileLayout>(n), seps};
}

std::optional<std::pair<TileLayout, std::vector<int64_t>>> TryGroup(
    TileLayout layout, const ffi::Array<PrimExpr>& shape) {
  // Same algorithm as Group but returns std::nullopt instead of ICHECK-failing
  // on regroup impossibility. Used by Apply(coord, shape) to opportunistically
  // pick the group-first path with a fallback to flatten+split.
  sym::Analyzer analyzer;
  size_t shape_idx = 0;
  PrimExpr prod = 1;

  std::vector<Iter> new_shard;
  std::vector<int64_t> seps{0};

  for (size_t i = 0; i < layout->shard.size(); ++i) {
    auto extent_i = layout->shard[i]->extent;
    auto stride_i = layout->shard[i]->stride;
    prod *= extent_i;
    while (shape_idx < shape.size() &&
           analyzer->CanProveEqual(floormod(prod, shape[shape_idx]), 0)) {
      PrimExpr c = analyzer->Simplify(floordiv(prod, shape[shape_idx]));
      if (!analyzer->CanProveEqual(floormod(extent_i, c), 0)) return std::nullopt;
      new_shard.push_back(Iter(analyzer->Simplify(floordiv(extent_i, c)),
                               analyzer->Simplify(stride_i * c), layout->shard[i]->axis));
      extent_i = c;
      prod = c;
      shape_idx++;
      seps.push_back(new_shard.size());
    }
    extent_i = analyzer->Simplify(extent_i);
    if (!IsOne(extent_i)) {
      if (shape_idx >= shape.size()) return std::nullopt;
      new_shard.push_back(Iter(extent_i, analyzer->Simplify(stride_i), layout->shard[i]->axis));
    }
  }

  if (shape_idx != shape.size()) return std::nullopt;

  auto* n = layout.CopyOnWrite();
  n->shard = new_shard;
  return std::make_pair(ffi::GetRef<TileLayout>(n), seps);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.TileLayoutGroup", [](const TileLayout& layout, const Array<PrimExpr>& shape) {
        auto [res, seps] = Group(layout, shape);
        return Tuple<TileLayout, Array<int64_t>>{res, Array<int64_t>(seps.begin(), seps.end())};
      });
}

Layout TileLayoutNode::Tile(const TileLayout& outer_in, const Array<PrimExpr>& outer_shape,
                            const Array<PrimExpr>& inner_shape) const {
  auto outer = outer_in->Canonicalize().as<TileLayout>().value();
  auto inner = ffi::GetRef<TileLayout>(this)->Canonicalize().as<TileLayout>().value();

  TVM_FFI_ICHECK_EQ(outer_shape.size(), inner_shape.size())
      << "Outer and inner shape size must match";

  auto [grouped_outer, outer_seps] = Group(outer, outer_shape);
  auto [grouped_inner, inner_seps] = Group(inner, inner_shape);

  outer = grouped_outer;
  inner = grouped_inner;

  sym::Analyzer analyzer;

  {
    // Scale outer axis strides by inner span on matching axes
    auto inner_span_map = BuildSpanMap(inner);
    std::vector<Iter> new_shard;
    for (size_t i = 0; i < outer->shard.size(); ++i) {
      auto it = inner_span_map.find(outer->shard[i]->axis.name());
      if (it != inner_span_map.end()) {
        new_shard.push_back(Iter(outer->shard[i]->extent, outer->shard[i]->stride * (*it).second,
                                 outer->shard[i]->axis));
      } else {
        new_shard.push_back(outer->shard[i]);
      }
    }
    outer = TileLayout(new_shard, outer->replica, outer->offset, outer->memory_origin);
  }

  TVM_FFI_ICHECK(!outer_seps.empty())
      << "Outer layout must only use split/reorder from logical scope";
  TVM_FFI_ICHECK(!inner_seps.empty())
      << "Inner layout must only use split/reorder from logical scope";

  std::vector<Iter> tile_shard;
  for (size_t i = 0; i < outer_shape.size(); ++i) {
    tile_shard.insert(tile_shard.end(), outer->shard.begin() + outer_seps[i],
                      outer->shard.begin() + outer_seps[i + 1]);

    tile_shard.insert(tile_shard.end(), inner->shard.begin() + inner_seps[i],
                      inner->shard.begin() + inner_seps[i + 1]);
  }

  std::vector<Iter> tile_rep{inner->replica.begin(), inner->replica.end()};
  tile_rep.insert(tile_rep.end(), outer->replica.begin(), outer->replica.end());

  ffi::Map<Axis, PrimExpr> tile_offset;
  for (const auto& [axis, off] : inner->offset) {
    tile_offset.Set(axis, off);
  }
  for (const auto& [axis, off] : outer->offset) {
    auto it = tile_offset.find(axis);
    if (it != tile_offset.end()) {
      tile_offset.Set(axis, (*it).second + off);
    } else {
      tile_offset.Set(axis, off);
    }
  }

  return TileLayout(tile_shard, tile_rep, tile_offset, outer->memory_origin + inner->memory_origin)
      ->Canonicalize();
}

}  // namespace tirx
}  // namespace tvm
