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

namespace tvm {
namespace relax {
namespace distributed {

StructInfo InferDistStructInfoSoftmax(const Call& call, const BlockBuilder& ctx) {
  Array<distributed::DTensorStructInfo> input_dtensor_sinfos = GetInputDTensorStructInfo(call, ctx);
  ICHECK(input_dtensor_sinfos.size() == 1);
  TensorStructInfo input_tensor_sinfo = input_dtensor_sinfos[0]->tensor_sinfo;

  if (input_tensor_sinfo->IsUnknownNdim()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Input of distributed operator must have known ndim");
  }
  if (!input_tensor_sinfo->IsUnknownDtype() && !input_tensor_sinfo->dtype.is_float()) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Softmax requires the input tensor to have float "
                                                "dtype. However, the given input dtype is "
                                             << input_tensor_sinfo->dtype);
  }
  const auto* attrs = call->attrs.as<SoftmaxAttrs>();
  NormalizeAxis(call, ctx, input_tensor_sinfo->ndim, attrs->axis);

  return InferShardingSpec(call, ctx, input_tensor_sinfo, distributed::BuildAxisGraphReduce);
}

TVM_REGISTER_OP("relax.nn.softmax")
    .set_attr<FInferStructInfo>("dist.FInferStructInfo", InferDistStructInfoSoftmax);

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
