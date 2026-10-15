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

namespace tvm {
namespace tirx {

// A changed type requires a fresh variable. Callers bind that variable to a
// decl_tensor over the source's physical pointer when using it as a view.
TVM_FFI_INLINE TensorVar FlattenedTensor(const TensorVar& tensor) {
  ffi::Array<PrimExpr> output_shape{1};
  if (tensor.strides().size()) {
    // Preserve the outer allocation footprint and cover every addressed element.
    // Permuted views can have their largest physical stride on an inner axis.
    TVM_FFI_ICHECK_EQ(tensor.shape().size(), tensor.strides().size());
    PrimExpr span = 1;
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
      span = span + (tensor.shape()[i] - 1) * tensor.strides()[i];
    }
    output_shape.Set(0, prim::Max(tensor.strides()[0] * tensor.shape()[0], span));
  } else {
    // Otherwise, the flattened extent is the product of the input extents.
    // This also flattens rank-0 tensors to a rank-1 tensor of shape [1].
    for (size_t i = 0; i < tensor.shape().size(); i++) {
      output_shape.Set(0, output_shape[0] * tensor.shape()[i]);
    }
  }

  if (output_shape.size() == tensor.shape().size() && tensor.strides().empty()) {
    return tensor;
  } else {
    // The original layout describes the old rank; reset it for the flattened shape.
    return TensorVar(
        tensor->name,
        TensorType(output_shape, tensor.dtype(), tensor.scope(), {}, tensor.elem_offset(),
                   tensor.ty()->data_alignment, tensor.ty()->offset_factor, std::nullopt),
        tensor->loc);
  }
}

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_TRANSFORM_FLATTENED_TENSOR_H_
