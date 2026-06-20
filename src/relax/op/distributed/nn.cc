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

#include "nn.h"

#include <tvm/ffi/extra/visit_error_context.h>

namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeSoftmax(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<distributed::DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  TVM_FFI_ICHECK(input_dtensor_tys.size() == 1);
  TensorType input_tensor_ty = input_dtensor_tys[0]->tensor_ty;

  if (input_tensor_ty->IsUnknownNdim()) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Input of distributed operator must have known ndim";
  }
  if (!input_tensor_ty->IsUnknownDtype() && !input_tensor_ty->dtype.is_float()) {
    TVM_FFI_VISIT_THROW(TypeError, call) << "Softmax requires the input tensor to have float "
                                            "dtype. However, the given input dtype is "
                                         << input_tensor_ty->dtype;
  }
  const auto* attrs = call->attrs.as<SoftmaxAttrs>();
  NormalizeAxis(call, ctx, input_tensor_ty->ndim, attrs->axis);

  return InferShardingSpec(call, ctx, input_tensor_ty, distributed::BuildAxisGraphReduce);
}

TVM_REGISTER_OP("relax.nn.softmax").set_attr<FInferType>("dist.FInferType", InferDistTypeSoftmax);

/* relax.nn.relu */
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(nn.relu, /*require_float_dtype=*/false);

/* relax.nn.gelu */
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(nn.gelu, /*require_float_dtype=*/true);

/* relax.nn.gelu_tanh */
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(nn.gelu_tanh, /*require_float_dtype=*/true);

/* relax.nn.silu */
RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(nn.silu, /*require_float_dtype=*/true);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
