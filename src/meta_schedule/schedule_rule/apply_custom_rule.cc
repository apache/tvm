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
#include <tvm/ffi/reflection/registry.h>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

class ApplyCustomRuleNode : public ScheduleRuleNode {
 public:
  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {
    CHECK(context->target.defined()) << "ValueError: Target is not defined in the tune context.";
    this->target_ = context->target;
  }

  static std::string GetCustomRuleName(const std::string& name, const std::string& key) {
    return "meta_schedule." + key + "." + name;
  }

  // Inherited from ScheduleRuleNode
  ffi::Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
    CHECK(this->target_.defined())
        << "ValueError: ApplyCustomRule is not initialized with TuneContext that has a Target.";
    ffi::Array<ffi::String> keys = this->target_.value()->keys;
    if (ffi::Optional<ffi::String> ann =
            tir::GetAnn<ffi::String>(sch->GetSRef(block_rv), "schedule_rule")) {
      if (ann.value() != "None") {
        for (const ffi::String& key : keys) {
          if (const auto custom_schedule_fn =
                  tvm::ffi::Function::GetGlobal(GetCustomRuleName(ann.value(), key))) {
            ffi::Array<tir::Schedule> result =
                (*custom_schedule_fn)(sch, block_rv).cast<ffi::Array<tir::Schedule>>();
            return result;
          }
        }
        std::ostringstream os;
        os << "Unknown schedule rule \"" << ann.value() << "\" for target keys \"" << keys
           << "\". Checked ffi::Functions:";
        for (const ffi::String& key : keys) {
          os << "\n  " << GetCustomRuleName(ann.value(), key);
        }
        LOG(WARNING) << os.str();
        sch->Unannotate(block_rv, "schedule_rule");
      }
    }
    return {sch};
  }

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<ApplyCustomRuleNode> n = ffi::make_object<ApplyCustomRuleNode>(*this);
    n->target_ = target_;
    return ScheduleRule(n);
  }

 public:
  ffi::Optional<Target> target_ = std::nullopt;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ApplyCustomRuleNode>().def_ro("target_", &ApplyCustomRuleNode::target_);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("meta_schedule.ApplyCustomRule", ApplyCustomRuleNode,
                                    ScheduleRuleNode);
};

ScheduleRule ScheduleRule::ApplyCustomRule() {
  ObjectPtr<ApplyCustomRuleNode> n = ffi::make_object<ApplyCustomRuleNode>();
  return ScheduleRule(n);
}

bool ScheduleRule::IsApplyCustomRule(const ScheduleRule& rule) {
  return rule->IsInstance<ApplyCustomRuleNode>();
}

TVM_FFI_STATIC_INIT_BLOCK() { ApplyCustomRuleNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("meta_schedule.ScheduleRuleApplyCustomRule", ScheduleRule::ApplyCustomRule);
}

}  // namespace meta_schedule
}  // namespace tvm
