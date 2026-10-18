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

/*!
 * \file flattened_tensor.h
 * \brief Physical tensor view construction for layout lowering.
 */
#ifndef TVM_TIRX_TRANSFORM_FLATTENED_TENSOR_H_
#define TVM_TIRX_TRANSFORM_FLATTENED_TENSOR_H_

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/layout.h>

#include <algorithm>

namespace tvm {
namespace tirx {

// SplitCoord leaves its leading quotient unbounded by the declared tile extent.
// Normalize that extent to the tensor domain before bounding or canonicalizing it.
TVM_FFI_INLINE TileLayout TileForTensorDomain(const TileLayout& tile,
                                              const ffi::Array<PrimExpr>& shape,
                                              const sym::Analyzer& analyzer) {
  if (tile->shard.empty() || std::any_of(shape.begin(), shape.end(), prim::IsZero)) return tile;
  PrimExpr size = IntImm(shape.empty() ? PrimType::Int(64) : shape[0].ty(), 1);
  for (size_t i = shape.size(); i > 0; --i) size = size * shape[i - 1];
  PrimExpr tail = IntImm(tile->shard.back()->extent.ty(), 1);
  for (size_t i = tile->shard.size(); i > 1; --i) tail = tail * tile->shard[i - 1]->extent;
  PrimExpr declared = tail * tile->shard[0]->extent;
  if (prim::ExprDeepEqual()(size, declared) || analyzer->CanProveEqual(size, declared)) return tile;
  auto shard = tile->shard;
  const auto& outer = shard[0];
  shard.Set(0,
            Iter(analyzer->Simplify(floordiv(size + tail - 1, tail)), outer->stride, outer->axis));
  return TileLayout(shard, tile->replica, tile->offset);
}

TVM_FFI_INLINE Layout LayoutForTensorDomain(const Layout& layout, const ffi::Array<PrimExpr>& shape,
                                            const sym::Analyzer& analyzer) {
  if (auto tile = layout.as<TileLayout>()) {
    return TileForTensorDomain(tile.value(), shape, analyzer);
  }
  if (auto compose = layout.as<ComposeLayout>()) {
    // A bare swizzle already applies to the entire tensor domain. Its tile
    // describes the swizzle period, rather than a bound on logical coordinates.
    if (compose.value()->tile_layout->IsTrivial()) return layout;
    auto result = compose.value();
    result.CopyOnWrite()->tile_layout = TileForTensorDomain(result->tile_layout, shape, analyzer);
    return result;
  }
  return layout;
}

// Relative flat indices are safe only when no stride can move before the origin.
TVM_FFI_INLINE bool HasNonnegativeStrides(const TileLayoutNode* tile,
                                          const sym::Analyzer& analyzer) {
  if (tile->IsContiguous(analyzer)) return true;
  return std::all_of(tile->shard.begin(), tile->shard.end(),
                     [&](const Iter& iter) { return analyzer->CanProve(iter->stride >= 0); });
}

TVM_FFI_INLINE PrimExpr TensorMemorySpan(const TensorVar& tensor, const sym::Analyzer& analyzer) {
  if (std::any_of(tensor.shape().begin(), tensor.shape().end(), prim::IsZero)) {
    return IntImm(tensor.ty()->DefaultIndexType(), 0);
  }
  Layout layout = LayoutForTensorDomain(tensor.ty()->layout.value(), tensor.shape(), analyzer);
  if (const auto* tile = layout.as<TileLayoutNode>()) {
    return analyzer->Simplify(tile->GetSpan(ffi::String("m")));
  }
  const auto* compose = layout.as_or_throw<ComposeLayout>().get();
  PrimExpr span = compose->tile_layout->GetSpan(ffi::String("m"));
  if (compose->tile_layout->IsTrivial()) {
    span = IntImm(tensor.ty()->DefaultIndexType(), 1);
    for (const auto& extent : tensor.shape()) span = span * extent;
  }
  if (compose->swizzle_len != 0) {
    int changed_bits = compose->per_element + compose->swizzle_len;
    if (!compose->swizzle_inner) changed_bits += compose->atom_len;
    PrimExpr block = IntImm(span.ty(), int64_t{1} << changed_bits);
    span = floordiv(span + block - 1, block) * block;
  }
  return analyzer->Simplify(span);
}

// A changed type requires a fresh variable. Callers bind that variable to a
// decl_tensor over the source's physical pointer when using it as a view.
TVM_FFI_INLINE TensorVar FlattenedTensor(const TensorVar& tensor,
                                         ffi::Optional<sym::Analyzer> analyzer = std::nullopt,
                                         bool fold_origin = false) {
  if (!analyzer.has_value()) analyzer = sym::Analyzer();
  const auto& ana = analyzer.value();
  ffi::Array<PrimExpr> output_shape{1};
  const auto* tile = tensor.ty()->layout.as<TileLayoutNode>();
  bool explicit_strides = tile && MatchShapeStridePattern(tensor.ty()).has_value();
  bool empty = std::any_of(tensor.shape().begin(), tensor.shape().end(), prim::IsZero);
  if (empty) {
    output_shape.Set(0, IntImm(tensor.ty()->DefaultIndexType(), 0));
  } else if (explicit_strides) {
    // Preserve the outer stride footprint while covering either stride direction.
    PrimExpr span = 1;
    for (const auto& iter : tile->shard) {
      span = span + (iter->extent - 1) * tvm::max(iter->stride, IntImm(iter->stride.ty(), 0));
    }
    if (tile->IsContiguous(ana)) {
      for (const auto& extent : tensor.shape()) output_shape.Set(0, output_shape[0] * extent);
    } else {
      output_shape.Set(0, tvm::max(tile->shard[0]->stride * tensor.shape()[0], span));
    }
    if (fold_origin || !HasNonnegativeStrides(tile, ana)) {
      // Absolute flat indices include the origin. A potentially reversed layout
      // also folds it, avoiding negative relative indices in the flattened view.
      static const Axis memory_axis = Axis::Get("m");
      if (auto origin = tile->offset.Get(memory_axis)) {
        output_shape.Set(0, output_shape[0] + origin.value());
      }
    }
  } else if (tile || tensor.ty()->layout.as<ComposeLayoutNode>()) {
    output_shape.Set(0, TensorMemorySpan(tensor, ana));
  } else {
    for (const auto& extent : tensor.shape()) output_shape.Set(0, output_shape[0] * extent);
  }

  if (output_shape.size() == tensor.shape().size() && !tensor.ty()->layout.has_value()) {
    return tensor;
  }
  return TensorVar(tensor->name,
                   TensorType(output_shape, tensor.dtype(), tensor.scope(),
                              tensor.ty()->data_alignment, std::nullopt),
                   tensor->loc);
}

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_TRANSFORM_FLATTENED_TENSOR_H_
