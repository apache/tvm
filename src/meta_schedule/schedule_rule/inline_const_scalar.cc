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

/*! \brief Inline blocks that produce a constant scalar. */
class InlineConstantScalarsNode : public ScheduleRuleNode {
 public:
  void InitializeWithTuneContext(const TuneContext& context) final {}

  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
    const std::string block_name = sch->Get(block_rv)->name_hint;
    if (block_name.find("compile_engine_const") != std::string::npos) {
      sch->ComputeInline(block_rv);
    }
    return {sch};
  }

  ScheduleRule Clone() const final {
    ObjectPtr<InlineConstantScalarsNode> n = make_object<InlineConstantScalarsNode>(*this);
    return ScheduleRule(n);
  }

  static constexpr const char* _type_key = "meta_schedule.InlineConstantScalars";
  TVM_DECLARE_FINAL_OBJECT_INFO(InlineConstantScalarsNode, ScheduleRuleNode);
};

ScheduleRule ScheduleRule::InlineConstantScalars() {
  ObjectPtr<InlineConstantScalarsNode> n = make_object<InlineConstantScalarsNode>();
  return ScheduleRule(n);
}

TVM_REGISTER_NODE_TYPE(InlineConstantScalarsNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleInlineConstantScalars")
    .set_body_typed(ScheduleRule::InlineConstantScalars);

}  // namespace meta_schedule
}  // namespace tvm
