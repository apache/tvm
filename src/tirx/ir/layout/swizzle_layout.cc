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

/**************** SwizzleLayout ****************/
SwizzleLayout::SwizzleLayout(int per_element, int swizzle_len, int atom_len, bool swizzle_inner) {
  auto n = ffi::make_object<SwizzleLayoutNode>();
  n->per_element = per_element;
  n->swizzle_len = swizzle_len;
  n->atom_len = atom_len;
  n->swizzle_inner = swizzle_inner;
  TVM_FFI_ICHECK(n->VerifyWellFormed()) << "ValueError: The swizzle layout is not well-formed";
  int swizzle_mask = (1 << swizzle_len) - 1;
  n->inner_mask = swizzle_mask;
  n->outer_mask = swizzle_mask << atom_len;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.SwizzleLayout",
                        [](int per_element, int swizzle_len, int atom_len, bool swizzle_inner) {
                          return SwizzleLayout(per_element, swizzle_len, atom_len, swizzle_inner);
                        });
}

bool SwizzleLayoutNode::CompatibleWithShape(const Array<PrimExpr>& shape) const { return true; }

bool SwizzleLayoutNode::VerifyWellFormed() const {
  return per_element >= 0 && swizzle_len >= 0 && atom_len >= swizzle_len;
}

PrimExpr SwizzleLayoutNode::GetSize(ffi::Optional<ffi::String> axis_name) const {
  TVM_FFI_ICHECK(!axis_name.has_value())
      << "ValueError: axis_name is not supported for swizzle layout";
  return 1 << (per_element + swizzle_len + atom_len);
}

PrimExpr SwizzleLayoutNode::GetSpan(ffi::Optional<ffi::String> axis_name) const {
  TVM_FFI_ICHECK(!axis_name.has_value())
      << "ValueError: axis_name is not supported for swizzle layout";
  return GetSize();
}

ffi::Map<ffi::String, PrimExpr> SwizzleLayoutNode::Apply(ffi::Array<PrimExpr> coord) const {
  LOG(FATAL) << "SwizzleLayoutNode::Apply(Array<PrimExpr>) is not implemented";
  return {};
}

ffi::Map<ffi::String, PrimExpr> SwizzleLayoutNode::Apply(PrimExpr coord) const {
  PrimExpr input = coord;
  auto f = [&](const PrimExpr& x) -> PrimExpr {
    if (swizzle_inner) {
      return x ^ ((x & outer_mask) >> atom_len);
    } else {
      return x ^ ((x & inner_mask) << atom_len);
    }
  };
  auto base = 1 << per_element;
  arith::Analyzer analyzer;
  // It takes more arithmetic operations to compute the result, but it is more friendly to the
  // vectorization. We use "m" as the default axis name here.
  return {
      {"m", analyzer.Simplify((f(floordiv(input, base)) << per_element) + floormod(input, base))}};
}

Layout SwizzleLayoutNode::Canonicalize() const { return ffi::GetRef<SwizzleLayout>(this); }

Layout SwizzleLayoutNode::Tile(const TileLayout& outer, const ffi::Array<PrimExpr>& outer_shape,
                               const ffi::Array<PrimExpr>& inner_shape) const {
  // Compose(Swizzle, Identity) -> then tile with `outer`.
  auto comp = ComposeLayout(ffi::GetRef<SwizzleLayout>(this), IdentityTileLayout(inner_shape));
  return comp->Tile(outer, outer_shape, inner_shape);
}

ffi::Optional<TileLayout> SwizzleLayoutNode::IsTileInner(
    const Layout& tile_layout, const ffi::Array<PrimExpr>& tiled_shape,
    const ffi::Array<PrimExpr>& inner_shape) const {
  // We expect tile_layout to be Compose(SwizzleLayout(this), _).
  if (auto comp = tile_layout.as<ComposeLayout>()) {
    if (StructuralEqual()(comp.value()->swizzle, ffi::GetRef<SwizzleLayout>(this))) {
      auto identity = IdentityTileLayout(inner_shape);
      return identity->IsTileInner(comp.value()->tile_layout, tiled_shape, inner_shape);
    }
  } else if (auto swizzle = tile_layout.as<SwizzleLayout>()) {
    if (StructuralEqual()(swizzle.value(), ffi::GetRef<SwizzleLayout>(this))) {
      auto inner_identity = IdentityTileLayout(inner_shape);
      auto tile_identity = IdentityTileLayout(tiled_shape);
      return inner_identity->IsTileInner(tile_identity, tiled_shape, inner_shape);
    }
  }
  return std::nullopt;
}

ffi::Optional<Layout> SwizzleLayoutNode::IsTileOuter(
    const Layout& tile_layout, const ffi::Array<PrimExpr>& tiled_shape,
    const ffi::Array<PrimExpr>& outer_shape) const {
  return std::nullopt;
}

ffi::Optional<Layout> SwizzleLayoutNode::Slice(const ffi::Array<PrimExpr>& shape,
                                               const Region& region) const {
  // Compose(Swizzle, Identity) -> then slice.
  auto comp = ComposeLayout(ffi::GetRef<SwizzleLayout>(this), IdentityTileLayout(shape));
  return comp->Slice(shape, region);
}

}  // namespace tirx
}  // namespace tvm
