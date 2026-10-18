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
 * \file flatten_tensor_utils.h
 * \brief Physical tensor view construction for layout lowering.
 */
#ifndef TVM_TIRX_TRANSFORM_FLATTEN_TENSOR_UTILS_H_
#define TVM_TIRX_TRANSFORM_FLATTEN_TENSOR_UTILS_H_

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/layout.h>

namespace tvm {
namespace tirx {

/*!
 * \brief Compute a conservative required backing-storage bound in elements.
 *
 * Bounds the memory axis "m" of a layout describing the actual tensor domain.
 * The bound includes stride gaps, memory-axis replicas, and memory-axis offsets;
 * it is neither a logical element count nor necessarily the minimum storage size.
 * Hardware-thread axes do not contribute, so distributed layouts give per-thread
 * storage. Swizzled layouts round up to whole blocks covering the bits the swizzle
 * may change, conservatively including padding.
 *
 * \param tensor Tensor with a TileLayout or ComposeLayout, valid nonnegative strides
 * and nonzero logical/tile extents. The layout must describe the tensor domain;
 * mismatched tile domains are not resized or extended. A bare swizzle describes
 * a period and uses the tensor shape for its domain. Borrowed views may have
 * nonzero memory offsets.
 * \param analyzer Analyzer used to simplify the bound.
 * \return Required storage bound measured from memory-axis zero, in tensor elements,
 * not bytes. This computes geometry only; it does not validate allocation policy.
 */
TVM_FFI_INLINE PrimExpr ComputeTensorMemorySize(const TensorVar& tensor,
                                                const sym::Analyzer& analyzer) {
  const Layout& layout = tensor.ty()->layout.value();
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

/*!
 * \brief Construct a one-dimensional physical tensor variable without allocating storage.
 *
 * The result preserves name, dtype, scope, alignment, and source location, and has
 * no layout. Its shape covers the physical footprint, including stride gaps and
 * conservative padding where needed. General layouts use ComputeTensorMemorySize,
 * including memory-axis offsets and excluding hardware-thread axes. Without a
 * layout, the shape is the logical element count.
 *
 * Callers bind a changed variable to a decl_tensor over the source pointer for a
 * borrowed view, and preserve any relative origin in that binding's geometry.
 * Allocation callers apply their own per-thread/backend storage contract; this
 * helper neither allocates nor rejects borrowed strided or offset views.
 *
 * \param tensor Tensor to flatten, assuming valid nonnegative strides and nonzero
 * logical/tile extents, with any layout describing the actual tensor domain.
 * \param analyzer Optional analyzer; a fresh analyzer is used when absent.
 * \param fold_origin For shape-aligned regular strides, include the memory origin
 * in the extent for absolute flat indexing. Otherwise the caller retains that
 * origin separately. General layouts already use an absolute bound regardless
 * of this flag.
 * \return Fresh physical tensor variable, or tensor itself if it is already
 * one-dimensional with no layout.
 */
TVM_FFI_INLINE TensorVar FlattenedTensor(const TensorVar& tensor,
                                         ffi::Optional<sym::Analyzer> analyzer = std::nullopt,
                                         bool fold_origin = false) {
  if (!analyzer.has_value()) analyzer = sym::Analyzer();
  const auto& ana = analyzer.value();
  ffi::Array<PrimExpr> output_shape{1};
  const auto* tile = tensor.ty()->layout.as<TileLayoutNode>();
  bool explicit_strides = tile && MatchShapeStridePattern(tensor.ty()).has_value();
  if (explicit_strides) {
    // Preserve the outer stride footprint and cover every addressed element.
    PrimExpr span = 1;
    for (const auto& iter : tile->shard) {
      span = span + (iter->extent - 1) * iter->stride;
    }
    if (tile->IsContiguous(ana)) {
      for (const auto& extent : tensor.shape()) output_shape.Set(0, output_shape[0] * extent);
    } else {
      output_shape.Set(0, tvm::max(tile->shard[0]->stride * tensor.shape()[0], span));
    }
    if (fold_origin) {
      // Absolute flat indices include the origin.
      static const Axis memory_axis = Axis::Get("m");
      if (auto origin = tile->offset.Get(memory_axis)) {
        output_shape.Set(0, output_shape[0] + origin.value());
      }
    }
  } else if (tile || tensor.ty()->layout.as<ComposeLayoutNode>()) {
    output_shape.Set(0, ComputeTensorMemorySize(tensor, ana));
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

#endif  // TVM_TIRX_TRANSFORM_FLATTEN_TENSOR_UTILS_H_
