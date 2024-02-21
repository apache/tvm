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

#include "utils.h"

namespace tvm {
namespace relax {
namespace distributed {

StructInfo InferDistStructInfoCallTIR(const Call& call, const BlockBuilder& ctx) {
  if (call->sinfo_args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "sinfo_args should have exact 1 output struct info.");
  }
  CHECK(call->args[0]->IsInstance<GlobalVarNode>())
      << "call_tir expects the first argument to be a GlobalVar referring to a TIR PrimFunc. "
      << "However, gets " << call->args[0];
  return call->sinfo_args[0];
}

TVM_REGISTER_OP("relax.call_tir")
    .set_attr<FInferStructInfo>("dist.FInferStructInfo", InferDistStructInfoCallTIR);

StructInfo InferDistStructInfoStopLiftParams(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call) << "stop_lift_params should have exact 1 arg.");
  }
  return Downcast<StructInfo>(call->args[0]->struct_info_.value());
}

TVM_REGISTER_OP("relax.builtin.stop_lift_params")
    .set_attr<FInferStructInfo>("dist.FInferStructInfo", InferDistStructInfoStopLiftParams);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
