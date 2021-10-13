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

ScheduleRule ScheduleRule::PyScheduleRule(
    PyScheduleRuleNode::FInitializeWithTuneContext f_initialize_with_tune_context,
    PyScheduleRuleNode::FApply f_apply) {
  ObjectPtr<PyScheduleRuleNode> n = make_object<PyScheduleRuleNode>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_apply = std::move(f_apply);
  return ScheduleRule(n);
}

TVM_REGISTER_OBJECT_TYPE(ScheduleRuleNode);
TVM_REGISTER_NODE_TYPE(PyScheduleRuleNode);

TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleInitializeWithTuneContext")
    .set_body_method<ScheduleRule>(&ScheduleRuleNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleApply")
    .set_body_method<ScheduleRule>(&ScheduleRuleNode::Apply);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRulePyScheduleRule")
    .set_body_typed(ScheduleRule::PyScheduleRule);

}  // namespace meta_schedule
}  // namespace tvm
