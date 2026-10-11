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
 * \file tensor_load.cc
 * \brief Validated tensor-load construction.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/expr.h>

namespace tvm {
namespace tirx {

// TensorLoad
TensorLoad MakeTensorLoad(TensorVar buffer, ffi::Array<PrimExpr> indices, Location loc) {
  TVM_FFI_ICHECK_EQ(buffer->shape.size(), indices.size())
      << "TensorVar " << buffer.name() << " is " << buffer->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
    TVM_FFI_ICHECK(indices[i].ty().IsScalar())
        << "Only the last index of a buffer access may be a vector type.";
  }

  PrimType result_ty = buffer->dtype;
  if (!indices.empty()) {
    PrimType index_ty = indices.back().ty();
    int16_t buffer_encoded_lanes = static_cast<int16_t>(buffer->dtype->dtype.lanes);
    bool is_buffer_dtype_scalable = buffer_encoded_lanes < -1;
    bool is_index_scalable = index_ty.IsScalableVector();

    TVM_FFI_ICHECK(!(is_index_scalable && is_buffer_dtype_scalable))
        << "Index dtype and buffer dtype can't both be scalable.";

    if (is_index_scalable) {
      result_ty = PrimType::ScalableVector(buffer->dtype.code(), buffer->dtype.bits(),
                                           index_ty.VScaleFactor() * buffer->dtype.lanes());
    } else if (is_buffer_dtype_scalable) {
      result_ty = PrimType::ScalableVector(buffer->dtype.code(), buffer->dtype.bits(),
                                           -buffer_encoded_lanes * index_ty.lanes());
    } else {
      result_ty = buffer->dtype.WithLanes(index_ty.lanes() * buffer->dtype.lanes());
    }
  }

  ffi::ObjectPtr<TensorLoadNode> node = ffi::make_object<TensorLoadNode>(std::move(buffer));
  node->ty = std::move(result_ty);
  node->indices = std::move(indices);
  node->loc = loc;
  return TensorLoad(std::move(node));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.TensorLoad",
                        [](TensorVar buffer, ffi::Array<PrimExpr> indices, Location loc) {
                          return MakeTensorLoad(buffer, indices, loc);
                        });
}

}  // namespace tirx
}  // namespace tvm
