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
#include <tvm/tir/transform.h>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

class VerifyVTCMLimitNode : public PostprocNode {
 public:
  Integer vtcm_capacity;

  void InitializeWithTuneContext(const TuneContext& context) final {
    ICHECK(context->target.defined());
    Target target = context->target.value();
    ICHECK(target->kind->name == "hexagon");
    // The value of 0 will disable VTCM verification.
    vtcm_capacity = target->GetAttr<Integer>("vtcm-capacity").value_or(0);
  }

  bool Verify(const IRModule& mod) const {
    if (!tir::VerifyVTCMLimit(mod, vtcm_capacity)) {
      return false;
    }
    return true;
  }

  bool Apply(const tir::Schedule& sch) final {
    IRModule mod = sch->mod();
    IRModule lowered{nullptr};
    auto pass_list = tir::GetVTCMCompactionPasses();
    transform::PassContext pass_ctx = transform::PassContext::Current();
    lowered = tvm::transform::Sequential(pass_list)(std::move(mod));
    if (!Verify(lowered)) {
      return false;
    }
    return true;
  }

  Postproc Clone() const {
    ObjectPtr<VerifyVTCMLimitNode> n = make_object<VerifyVTCMLimitNode>(*this);
    return Postproc(n);
  }

  static constexpr const char* _type_key = "meta_schedule.VerifyVTCMLimit";
  TVM_DECLARE_FINAL_OBJECT_INFO(VerifyVTCMLimitNode, PostprocNode);
};

Postproc Postproc::VerifyVTCMLimit() {
  ObjectPtr<VerifyVTCMLimitNode> n = make_object<VerifyVTCMLimitNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(VerifyVTCMLimitNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocVerifyVTCMLimit")
    .set_body_typed(Postproc::VerifyVTCMLimit);

}  // namespace meta_schedule
}  // namespace tvm
