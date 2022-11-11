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
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
    CHECK(this->target_.defined())
        << "ValueError: ApplyCustomRule is not initialized with TuneContext that has a Target.";
    Array<String> keys = this->target_.value()->keys;
    if (Optional<String> ann = tir::GetAnn<String>(sch->GetSRef(block_rv), "schedule_rule")) {
      if (ann.value() != "None") {
        for (const String& key : keys) {
          if (const runtime::PackedFunc* custom_schedule_fn =
                  runtime::Registry::Get(GetCustomRuleName(ann.value(), key))) {
            Array<tir::Schedule> result = ((*custom_schedule_fn)(sch, block_rv));
            return result;
          }
        }
        std::ostringstream os;
        os << "Unknown schedule rule \"" << ann.value() << "\" for target keys \"" << keys
           << "\". Checked PackedFuncs:";
        for (const String& key : keys) {
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
    ObjectPtr<ApplyCustomRuleNode> n = make_object<ApplyCustomRuleNode>(*this);
    n->target_ = target_;
    return ScheduleRule(n);
  }

 public:
  Optional<Target> target_ = NullOpt;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("target_", &target_); }

  static constexpr const char* _type_key = "meta_schedule.ApplyCustomRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ApplyCustomRuleNode, ScheduleRuleNode);
};

ScheduleRule ScheduleRule::ApplyCustomRule() {
  ObjectPtr<ApplyCustomRuleNode> n = make_object<ApplyCustomRuleNode>();
  return ScheduleRule(n);
}

bool ScheduleRule::IsApplyCustomRule(const ScheduleRule& rule) {
  return rule->IsInstance<ApplyCustomRuleNode>();
}

TVM_REGISTER_NODE_TYPE(ApplyCustomRuleNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleApplyCustomRule")
    .set_body_typed(ScheduleRule::ApplyCustomRule);

}  // namespace meta_schedule
}  // namespace tvm
