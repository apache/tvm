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
ComposeLayout::ComposeLayout(SwizzleLayout layout_A, TileLayout layout_B) {
  auto n = ffi::make_object<ComposeLayoutNode>();
  n->swizzle = layout_A;
  n->tile_layout = layout_B;
  TVM_FFI_ICHECK(n->VerifyWellFormed()) << "ValueError: The compose layout is not well-formed";

  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.ComposeLayout", [](SwizzleLayout layout_A, TileLayout layout_B) {
    return ComposeLayout(layout_A, layout_B);
  });
}

bool ComposeLayoutNode::CompatibleWithShape(const Array<PrimExpr>& shape) const { return true; }

bool ComposeLayoutNode::VerifyWellFormed() const {
  if (!swizzle->VerifyWellFormed() || !tile_layout->VerifyWellFormed()) {
    return false;
  }
  return true;
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
  auto swizzle_res = swizzle->Apply(m);
  TVM_FFI_ICHECK(swizzle_res.size() == 1 && swizzle_res.find("m") != swizzle_res.end());
  return swizzle_res;
}

Layout ComposeLayoutNode::Canonicalize() const {
  auto tile_normalized = tile_layout->Canonicalize().as<TileLayout>().value();
  if (tile_normalized->IsTrivial()) {
    return swizzle;
  }
  return ComposeLayout(swizzle, tile_normalized);
}

Layout ComposeLayoutNode::Tile(const TileLayout& outer, const ffi::Array<PrimExpr>& outer_shape,
                               const ffi::Array<PrimExpr>& inner_shape) const {
  // layout_B is first tiled with `outer`, then compose with layout_A.
  auto tiled_B = tile_layout->Tile(outer, outer_shape, inner_shape).as<TileLayout>().value();
  return ComposeLayout(swizzle, tiled_B);
}

ffi::Optional<TileLayout> ComposeLayoutNode::IsTileInner(
    const Layout& tile_layout, const ffi::Array<PrimExpr>& tiled_shape,
    const ffi::Array<PrimExpr>& inner_shape) const {
  if (auto comp = tile_layout.as<ComposeLayout>()) {
    if (StructuralEqual()(comp.value()->swizzle, this->swizzle)) {
      return this->tile_layout->IsTileInner(comp.value()->tile_layout, tiled_shape, inner_shape);
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
  // Slice applies to the tile layout then compose with swizzle.
  auto sliced_opt = tile_layout->Slice(shape, region);
  if (!sliced_opt.has_value()) return std::nullopt;
  auto sliced = sliced_opt.value().as<TileLayout>().value();
  return ComposeLayout(swizzle, sliced);
}

}  // namespace tirx
}  // namespace tvm
