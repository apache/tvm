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

void PyScheduleRuleNode::InitializeWithTuneContext(const TuneContext& context) {
  ICHECK(f_initialize_with_tune_context != nullptr)
      << "PyScheduleRule's InitializeWithTuneContext method not implemented!";
  f_initialize_with_tune_context(context);
}

Array<tir::Schedule> PyScheduleRuleNode::Apply(const tir::Schedule& sch,
                                               const tir::BlockRV& block) {
  ICHECK(f_apply != nullptr) << "PyScheduleRule's Apply method not implemented!";
  return f_apply(sch, block);
}

ScheduleRule PyScheduleRuleNode::Clone() const {
  ICHECK(f_clone != nullptr) << "PyScheduleRule's Clone method not implemented!";
  return f_clone();
}

ScheduleRule ScheduleRule::PyScheduleRule(
    PyScheduleRuleNode::FInitializeWithTuneContext f_initialize_with_tune_context,  //
    PyScheduleRuleNode::FApply f_apply,                                             //
    PyScheduleRuleNode::FClone f_clone,                                             //
    PyScheduleRuleNode::FAsString f_as_string) {
  ObjectPtr<PyScheduleRuleNode> n = make_object<PyScheduleRuleNode>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_apply = std::move(f_apply);
  n->f_clone = std::move(f_clone);
  n->f_as_string = std::move(f_as_string);
  return ScheduleRule(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PyScheduleRuleNode>([](const ObjectRef& n, ReprPrinter* p) {
      const auto* self = n.as<PyScheduleRuleNode>();
      ICHECK(self);
      PyScheduleRuleNode::FAsString f_as_string = (*self).f_as_string;
      ICHECK(f_as_string != nullptr) << "PyScheduleRule's AsString method not implemented!";
      p->stream << f_as_string();
    });

TVM_REGISTER_OBJECT_TYPE(ScheduleRuleNode);
TVM_REGISTER_NODE_TYPE(PyScheduleRuleNode);

TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleInitializeWithTuneContext")
    .set_body_method<ScheduleRule>(&ScheduleRuleNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleApply")
    .set_body_method<ScheduleRule>(&ScheduleRuleNode::Apply);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleClone")
    .set_body_method<ScheduleRule>(&ScheduleRuleNode::Clone);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRulePyScheduleRule")
    .set_body_typed(ScheduleRule::PyScheduleRule);

}  // namespace meta_schedule
}  // namespace tvm
