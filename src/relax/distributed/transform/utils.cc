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

bool SinfoCompatibleWithDistIR(Array<StructInfo> sinfos) {
  bool compatible = true;
  for (const auto& sinfo : sinfos) {
    if (const auto* tuple_sinfo = sinfo.as<TupleStructInfoNode>()) {
      compatible &= SinfoCompatibleWithDistIR(tuple_sinfo->fields);
    } else {
      compatible &= !sinfo->IsInstance<TensorStructInfoNode>();
    }
  }
  return compatible;
}

bool SinfoCompatibleWithRelax(Array<StructInfo> sinfos) {
  bool compatible = true;
  for (const auto& sinfo : sinfos) {
    if (const auto* tuple_sinfo = sinfo.as<TupleStructInfoNode>()) {
      compatible &= SinfoCompatibleWithRelax(tuple_sinfo->fields);
    } else {
      compatible &= !sinfo->IsInstance<DTensorStructInfoNode>();
    }
  }
  return compatible;
}
bool IsDistIRFunc(Function func) {
  Array<StructInfo> param_sinfos;
  for (const auto& param : func->params) {
    ICHECK(param->struct_info_);
    param_sinfos.push_back(Downcast<StructInfo>(param->struct_info_.value()));
  }
  bool compatible_with_dist_ir = SinfoCompatibleWithDistIR(param_sinfos);
  bool compatible_with_relax = SinfoCompatibleWithRelax(param_sinfos);
  if (compatible_with_relax) {
    return false;
  } else if (compatible_with_dist_ir && !compatible_with_relax) {
    return true;
  } else {
    LOG(FATAL) << "mixed use of DTensor and Tensor in: " << func;
  }
}

bool IsShardingAnnotatedFunc(Function func) {
  bool has_annotate_sharding = false;
  PostOrderVisit(func, [&has_annotate_sharding](const Expr& e) {
    const CallNode* call = e.as<CallNode>();
    if (!call) {
      return;
    }
    static Op annotate_sharding_op = Op::Get("relax.dist.annotate_sharding");
    if (call->op.same_as(annotate_sharding_op)) {
      has_annotate_sharding = true;
    }
  });
  return has_annotate_sharding;
}
}  // namespace distributed
}  // namespace relax
}  // namespace tvm
