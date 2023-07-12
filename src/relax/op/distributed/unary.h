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
 * \file unary.h
 * \brief The functions to infer struct info for distributed unary operator
 */

#ifndef TVM_RELAX_OP_DISTRIBUTED_UNARY_H_
#define TVM_RELAX_OP_DISTRIBUTED_UNARY_H_

#include "utils.h"

namespace tvm {
namespace relax {
namespace distributed {

template <bool require_float_dtype, typename FType>
StructInfo InferDistStructInfoUnary(const Call& call, const BlockBuilder& ctx,
                                    FType f_compute_out_dtype) {
  Array<distributed::DTensorStructInfo> input_dtensor_sinfos = GetInputDTensorStructInfo(call, ctx);
  ICHECK(input_dtensor_sinfos.size() == 1);
  distributed::DTensorStructInfo input_dtensor_sinfo = input_dtensor_sinfos[0];
  TensorStructInfo input_tensor_sinfo = input_dtensor_sinfo->tensor_sinfo;

  if (require_float_dtype && !input_tensor_sinfo->IsUnknownDtype() &&
      !input_tensor_sinfo->dtype.is_float()) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << call->op
        << " requires the input tensor to have float dtype. However, the given input dtype is "
        << input_tensor_sinfo->dtype);
  }
  auto output_sinfo = make_object<TensorStructInfoNode>(*input_tensor_sinfo.get());
  output_sinfo->dtype = f_compute_out_dtype(input_tensor_sinfo);
  TensorStructInfo out_tensor_sinfo(output_sinfo);
  return distributed::DTensorStructInfo(out_tensor_sinfo, input_dtensor_sinfo->device_mesh,
                                        input_dtensor_sinfo->placement);
}

template <bool require_float_dtype>
StructInfo InferDistStructInfoUnaryArith(const Call& call, const BlockBuilder& ctx) {
  return InferDistStructInfoUnary<require_float_dtype>(
      call, ctx, [](const TensorStructInfo& input_sinfo) { return input_sinfo->dtype; });
}

StructInfo InferDistStructInfoUnaryCheck(const Call& call, const BlockBuilder& ctx);

#define RELAX_REGISTER_UNARY_ARITH_DIST_INFER_STRUCT_INFO(OpName, RequireFloatDtype) \
  TVM_REGISTER_OP("relax." #OpName)                                                  \
      .set_attr<FInferStructInfo>("dist.FInferStructInfo",                           \
                                  InferDistStructInfoUnaryArith<RequireFloatDtype>)

#define RELAX_REGISTER_UNARY_CHECK_DIST_INFER_STRUCT_INFO(OpName) \
  TVM_REGISTER_OP("relax." #OpName)                               \
      .set_attr<FInferStructInfo>("dist.FInferStructInfo", InferDistStructInfoUnaryCheck)

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_DISTRIBUTED_UNARY_H_
