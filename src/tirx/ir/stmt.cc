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
 * \file tvm/tirx/stmt.cc
 */
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/stmt.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>

#include <iterator>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {
namespace {
TVM_FFI_INLINE int GetLanesOrVScaleFactor(const PrimType& ty) {
  return ty.IsScalableVector() ? ty.VScaleFactor() : ty.lanes();
}

void ValidateTensorStore(Expr dest, ffi::Array<PrimExpr> indices, PrimExpr value) {
  TensorType type = dest->ty.as_or_throw<TensorType>();
  TVM_FFI_ICHECK_EQ(type->shape.size(), indices.size())
      << "Destination " << dest << " is " << type->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
    TVM_FFI_ICHECK(indices[i].ty().IsScalar())
        << "Only the last index of a tensor access may be a vector type.";
  }

  bool is_index_scalable = indices.empty() ? false : indices.back().ty().IsScalableVector();
  int16_t tensor_encoded_lanes = static_cast<int16_t>(type->dtype->dtype.lanes);
  bool is_tensor_dtype_scalable = tensor_encoded_lanes < -1;
  PrimType value_ty = value.ty();
  bool is_value_dtype_scalable = value_ty.IsScalableVector();

  TVM_FFI_ICHECK(!(is_index_scalable && is_tensor_dtype_scalable))
      << "Index dtype and tensor dtype can't both be scalable.";

  if (is_index_scalable || is_tensor_dtype_scalable) {
    TVM_FFI_ICHECK(is_value_dtype_scalable) << "Can't store non-scalable data into scalable tensor";
  }

  int index_lanes = indices.empty() ? 1 : GetLanesOrVScaleFactor(indices.back().ty());
  int tensor_lanes = is_tensor_dtype_scalable ? -tensor_encoded_lanes : tensor_encoded_lanes;
  int value_dtype_lanes = GetLanesOrVScaleFactor(value_ty);

  TVM_FFI_ICHECK_EQ(index_lanes * tensor_lanes, value_dtype_lanes)
      << "Cannot store value with " << value_dtype_lanes << ", expected value with "
      << index_lanes * tensor_lanes << " (" << index_lanes << " index lanes * " << tensor_lanes
      << " tensor element lanes)";

  PrimType tensor_dtype = PrimType::Void();
  if (is_index_scalable || is_tensor_dtype_scalable) {
    tensor_dtype = PrimType::ScalableVector(type->dtype.code(), type->dtype.bits(),
                                            tensor_lanes * index_lanes);
  } else {
    tensor_dtype = type->dtype.WithLanes(tensor_lanes * index_lanes);
  }
  if (tensor_dtype != value_ty) {
    TVM_FFI_THROW(TypeError) << "dtype mismatch on TensorStore: "                 //
                             << "tensor's dtype is `" << type->dtype              //
                             << "`, the lanes of indexing are: `" << index_lanes  //
                             << "`, the scalability is: `" << tensor_dtype.IsScalableVector()
                             << "`, but RHS's dtype is `" << value_ty << "`";
  }
}

void ValidateTensorEvaluate(Expr value) {
  TVM_FFI_ICHECK(!value->IsInstance<VarNode>())
      << "A tensor variable cannot be used as a scalar Evaluate value; "
      << "use tensor.data to evaluate its physical pointer";
}
}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::TypeAttrDef<TensorTypeNode>()
      .def(tvm::type_attr::kTensorStoreValidate, ValidateTensorStore)
      .def(tvm::type_attr::kEvaluateValidate, ValidateTensorEvaluate);
}

}  // namespace tirx
}  // namespace tvm
