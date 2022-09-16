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

void PySpaceGeneratorNode::InitializeWithTuneContext(const TuneContext& context) {
  ICHECK(f_initialize_with_tune_context != nullptr)
      << "PySpaceGenerator's InitializeWithTuneContext method not implemented!";
  f_initialize_with_tune_context(context);
}

Array<tir::Schedule> PySpaceGeneratorNode::GenerateDesignSpace(const IRModule& mod) {
  ICHECK(f_generate_design_space != nullptr)
      << "PySpaceGenerator's GenerateDesignSpace method not implemented!";
  return f_generate_design_space(mod);
}

SpaceGenerator PySpaceGeneratorNode::Clone() const {
  ICHECK(f_clone != nullptr) << "PySpaceGenerator's Clone method not implemented!";
  return f_clone();
}

SpaceGenerator SpaceGenerator::PySpaceGenerator(
    FInitializeWithTuneContext f_initialize_with_tune_context,
    FGenerateDesignSpace f_generate_design_space, FClone f_clone) {
  ObjectPtr<PySpaceGeneratorNode> n = make_object<PySpaceGeneratorNode>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_generate_design_space = std::move(f_generate_design_space);
  n->f_clone = std::move(f_clone);
  return SpaceGenerator(n);
}

TVM_REGISTER_OBJECT_TYPE(SpaceGeneratorNode);
TVM_REGISTER_NODE_TYPE(PySpaceGeneratorNode);

TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorInitializeWithTuneContext")
    .set_body_method<SpaceGenerator>(&SpaceGeneratorNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorGenerateDesignSpace")
    .set_body_method<SpaceGenerator>(&SpaceGeneratorNode::GenerateDesignSpace);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorPySpaceGenerator")
    .set_body_typed(SpaceGenerator::PySpaceGenerator);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorClone")
    .set_body_method<SpaceGenerator>(&SpaceGeneratorNode::Clone);

}  // namespace meta_schedule
}  // namespace tvm
