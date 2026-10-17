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
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>

#include <memory>

#include "utils.h"

namespace tvm {
namespace tirx {

bool IsContiguous(const ffi::Optional<Layout>& layout) {
  if (!layout.has_value()) return true;
  const auto* tile = layout.as<TileLayoutNode>();
  if (!tile || !tile->replica.empty()) return false;
  static const Axis memory_axis = Axis::Get("m");
  for (const auto& [axis, offset] : tile->offset) {
    if (axis != memory_axis) return false;
  }
  prim::ExprDeepEqual deep_equal;
  std::unique_ptr<sym::Analyzer> analyzer;
  auto equal = [&](const PrimExpr& lhs, const PrimExpr& rhs) {
    if (deep_equal(lhs, rhs)) return true;
    if (!analyzer) analyzer = std::make_unique<sym::Analyzer>();
    return (*analyzer)->CanProveEqual(lhs, rhs);
  };
  PrimExpr stride =
      IntImm(tile->shard.empty() ? PrimType::Int(64) : tile->shard.back()->stride.ty(), 1);
  for (size_t i = tile->shard.size(); i > 0; --i) {
    const auto& iter = tile->shard[i - 1];
    if (iter->axis != memory_axis) return false;
    if (!prim::IsOne(iter->extent) && !equal(iter->stride, stride)) return false;
    stride = stride * iter->extent;
  }
  return true;
}

/**************** Layout ****************/
ffi::Map<ffi::String, PrimExpr> LayoutNode::Apply(const ffi::Array<PrimExpr>& coord,
                                                  const ffi::Array<PrimExpr>& shape) const {
  TVM_FFI_ICHECK_EQ(coord.size(), shape.size())
      << "ValueError: The size of coord and shape should be equal";
  return Apply(FlattenCoord(coord, shape));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  auto def = refl::GlobalDef();
  def.def("tirx.IsContiguous", IsContiguous);
  def.def("tirx.LayoutVerifyWellFormed", [](Layout layout) { return layout->VerifyWellFormed(); });
  def.def("tirx.LayoutGetSize", [](Layout layout, ffi::Optional<ffi::String> axis_name) {
    return layout->GetSize(axis_name);
  });
  def.def("tirx.LayoutGetSpan", [](Layout layout, ffi::Optional<ffi::String> axis_name) {
    return layout->GetSpan(axis_name);
  });
  def.def("tirx.LayoutApplyWithShape",
          [](Layout layout, ffi::Array<PrimExpr> coord, ffi::Array<PrimExpr> shape) {
            return layout->Apply(coord, shape);
          });
  def.def("tirx.LayoutApply",
          [](Layout layout, ffi::Array<PrimExpr> coord) { return layout->Apply(coord); });
  def.def("tirx.LayoutApplyLinear",
          [](Layout layout, PrimExpr coord) { return layout->Apply(coord); });
  def.def("tirx.LayoutCanonicalize", [](Layout layout) { return layout->Canonicalize(); });
  def.def("tirx.LayoutTile", [](Layout layout, TileLayout outer, ffi::Array<PrimExpr> outer_shape,
                                ffi::Array<PrimExpr> inner_shape) {
    return layout->Tile(outer, outer_shape, inner_shape);
  });
}

}  // namespace tirx
}  // namespace tvm
