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
 * \file binary.h
 * \brief The functions to infer struct info for distributed binary operator
 */

#ifndef TVM_RELAX_OP_DISTRIBUTED_BINARY_H_
#define TVM_RELAX_OP_DISTRIBUTED_BINARY_H_

#include <algorithm>

#include "utils.h"

namespace tvm {
namespace relax {
namespace distributed {

template <typename FType>
StructInfo InferDistStructInfoBroadcast(const Call& call, const BlockBuilder& ctx,
                                        FType f_compute_out_dtype) {
  Array<distributed::DTensorStructInfo> input_dtensor_sinfos = GetInputDTensorStructInfo(call, ctx);
  TensorStructInfo x1_sinfo, x2_sinfo;
  x1_sinfo = input_dtensor_sinfos[0]->tensor_sinfo;
  x2_sinfo = input_dtensor_sinfos[1]->tensor_sinfo;

  // DateType
  DataType output_dtype = f_compute_out_dtype(call, ctx, x1_sinfo, x2_sinfo);

  // ndims
  ICHECK(!x1_sinfo->IsUnknownNdim() && !x2_sinfo->IsUnknownNdim())
      << "Unknown ndim is not supported for distributed operators.";
  int output_ndim = std::max(x1_sinfo->ndim, x2_sinfo->ndim);

  const auto* x1_shape = x1_sinfo->shape.as<ShapeExprNode>();
  const auto* x2_shape = x2_sinfo->shape.as<ShapeExprNode>();
  TensorStructInfo output_tensor_sinfo;
  // Shapes and ndims
  if (x1_shape && x2_shape) {
    // If all inputs have shapes, directly infer shapes
    Optional<Array<PrimExpr>> output_shape =
        InferBinaryBroadcastShape(call, ctx, x1_shape->values, x2_shape->values);
    if (!output_shape.defined()) {
      output_tensor_sinfo = TensorStructInfo(output_dtype, /*ndim=*/output_ndim);
    } else {
      ICHECK_EQ(static_cast<int>(output_shape.value().size()), output_ndim);
      output_tensor_sinfo = TensorStructInfo(ShapeExpr(output_shape.value()), output_dtype);
    }
  } else {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Cannot infer shape for binary broadcast operator.");
  }
  return InferShardingSpec(call, ctx, output_tensor_sinfo, distributed::BuildAxisGraphBinary);
}

StructInfo InferDistStructInfoBroadcastArith(const Call& call, const BlockBuilder& ctx);

StructInfo InferDistStructInfoBroadcastCMP(const Call& call, const BlockBuilder& ctx);

#define RELAX_REGISTER_BINARY_BROADCAST_DIST_INFER_STRUCT_INFO(OpName) \
  TVM_REGISTER_OP("relax." #OpName)                                    \
      .set_attr<FInferStructInfo>("dist.FInferStructInfo", InferDistStructInfoBroadcastArith)

#define RELAX_REGISTER_CMP_DIST_INFER_STRUCT_INFO(OpName) \
  TVM_REGISTER_OP("relax." #OpName)                       \
      .set_attr<FInferStructInfo>("dist.FInferStructInfo", InferDistStructInfoBroadcastCMP)

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_DISTRIBUTED_BINARY_H_
