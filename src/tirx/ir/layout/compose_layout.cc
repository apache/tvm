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
#include "utils.h"

namespace tvm {
namespace tirx {

/**************** ComposeLayout ****************/
ComposeLayout::ComposeLayout(int per_element, int swizzle_len, int atom_len, TileLayout tile_layout,
                             bool swizzle_inner) {
  auto n = ffi::make_object<ComposeLayoutNode>();
  n->per_element = per_element;
  n->swizzle_len = swizzle_len;
  n->atom_len = atom_len;
  n->swizzle_inner = swizzle_inner;
  n->tile_layout = std::move(tile_layout);
  TVM_FFI_ICHECK(n->VerifyWellFormed()) << "ValueError: The compose layout is not well-formed";
  int swizzle_mask = (1 << swizzle_len) - 1;
  n->inner_mask = swizzle_mask;
  n->outer_mask = swizzle_mask << atom_len;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.ComposeLayout", [](int per_element, int swizzle_len, int atom_len,
                                                 TileLayout tile_layout, bool swizzle_inner) {
    return ComposeLayout(per_element, swizzle_len, atom_len, tile_layout, swizzle_inner);
  });
}

bool ComposeLayoutNode::CompatibleWithShape(const Array<PrimExpr>& shape) const { return true; }

bool ComposeLayoutNode::VerifyWellFormed() const {
  if (!(per_element >= 0 && swizzle_len >= 0 && atom_len >= swizzle_len)) {
    return false;
  }
  return tile_layout->VerifyWellFormed();
}

PrimExpr ComposeLayoutNode::GetSize(ffi::Optional<ffi::String> axis_name) const {
  TVM_FFI_ICHECK(!axis_name.has_value())
      << "ValueError: axis_name is not supported for compose layout";
  return tile_layout->GetSize(axis_name);
}

PrimExpr ComposeLayoutNode::GetSpan(ffi::Optional<ffi::String> axis_name) const {
  TVM_FFI_ICHECK(!axis_name.has_value())
      << "ValueError: axis_name is not supported for compose layout";
  return tile_layout->GetSpan(axis_name);
}

ffi::Map<ffi::String, PrimExpr> ComposeLayoutNode::Apply(ffi::Array<PrimExpr> coord) const {
  LOG(FATAL) << "ComposeLayoutNode::Apply(Array<PrimExpr>) is not implemented";
  return {};
}

ffi::Map<ffi::String, PrimExpr> ComposeLayoutNode::Apply(PrimExpr coord) const {
  auto res = tile_layout->Apply(coord);
  TVM_FFI_ICHECK(res.size() == 1 && res.find("m") != res.end());
  auto m = res["m"];
  // Inline the swizzle XOR (formerly SwizzleLayoutNode::Apply): the swizzle
  // operates on the tile-mapped coordinate ``m``.
  auto f = [&](const PrimExpr& x) -> PrimExpr {
    if (swizzle_inner) {
      return x ^ ((x & outer_mask) >> atom_len);
    } else {
      return x ^ ((x & inner_mask) << atom_len);
    }
  };
  auto base = 1 << per_element;
  arith::Analyzer analyzer;
  return {{"m", analyzer->Simplify((f(floordiv(m, base)) << per_element) + floormod(m, base))}};
}

Layout ComposeLayoutNode::Canonicalize() const {
  auto tile_normalized = tile_layout->Canonicalize().as<TileLayout>().value();
  return ComposeLayout(per_element, swizzle_len, atom_len, tile_normalized, swizzle_inner);
}

Layout ComposeLayoutNode::Tile(const TileLayout& outer, const ffi::Array<PrimExpr>& outer_shape,
                               const ffi::Array<PrimExpr>& inner_shape) const {
  // A bare swizzle (ComposeLayout with a trivial tile) carries only the swizzle
  // period, not a tile matching `inner_shape`; substitute an identity tile over
  // the inner product first, matching the former SwizzleLayoutNode::Tile.
  TileLayout base = this->tile_layout;
  if (base->IsTrivial()) {
    base = IdentityTileLayout(inner_shape);
  }
  auto tiled_B = base->Tile(outer, outer_shape, inner_shape).as<TileLayout>().value();
  return ComposeLayout(per_element, swizzle_len, atom_len, tiled_B, swizzle_inner);
}

ffi::Optional<TileLayout> ComposeLayoutNode::IsTileInner(
    const Layout& tile_layout, const ffi::Array<PrimExpr>& tiled_shape,
    const ffi::Array<PrimExpr>& inner_shape) const {
  if (auto comp = tile_layout.as<ComposeLayout>()) {
    if (comp.value()->per_element == this->per_element &&
        comp.value()->swizzle_len == this->swizzle_len &&
        comp.value()->atom_len == this->atom_len &&
        comp.value()->swizzle_inner == this->swizzle_inner) {
      // A bare swizzle (ComposeLayout with a trivial tile) has no real tile to
      // compare; its "tile" is the identity over the inner product, and a bare
      // `tile_layout` argument contributes the identity over the tiled product.
      // Substitute those identities so TileLayoutNode::IsTileInner sees the same
      // inputs the former SwizzleLayoutNode::IsTileInner produced.
      TileLayout this_tile = this->tile_layout;
      if (this->tile_layout->IsTrivial()) {
        this_tile = IdentityTileLayout(inner_shape);
      }
      TileLayout arg_tile = comp.value()->tile_layout;
      if (comp.value()->tile_layout->IsTrivial()) {
        arg_tile = IdentityTileLayout(tiled_shape);
      }
      return this_tile->IsTileInner(arg_tile, tiled_shape, inner_shape);
    }
  }
  return std::nullopt;
}

ffi::Optional<Layout> ComposeLayoutNode::IsTileOuter(
    const Layout& tile_layout, const ffi::Array<PrimExpr>& tiled_shape,
    const ffi::Array<PrimExpr>& outer_shape) const {
  return std::nullopt;
}

ffi::Optional<Layout> ComposeLayoutNode::Slice(const ffi::Array<PrimExpr>& shape,
                                               const Region& region) const {
  // A bare swizzle (ComposeLayout with a trivial tile) carries only the swizzle
  // period, not a tile matching `shape`; substitute an identity tile over the
  // buffer shape first, matching the former SwizzleLayoutNode::Slice.
  TileLayout base = this->tile_layout;
  if (base->IsTrivial()) {
    base = IdentityTileLayout(shape);
  }
  auto sliced_opt = base->Slice(shape, region);
  if (!sliced_opt.has_value()) return std::nullopt;
  auto sliced = sliced_opt.value().as<TileLayout>().value();
  return ComposeLayout(per_element, swizzle_len, atom_len, sliced, swizzle_inner);
}

}  // namespace tirx
}  // namespace tvm
