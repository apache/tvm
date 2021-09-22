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
#include <tvm/meta_schedule/utils.h>

namespace tvm {
namespace meta_schedule {

SpaceGenerator SpaceGenerator::PySpaceGenerator(
    PySpaceGeneratorNode::FInitializeWithTuneContext f_initialize_with_tune_context,
    PySpaceGeneratorNode::FGenerateDesignSpace f_generate_design_space) {
  ObjectPtr<PySpaceGeneratorNode> n = make_object<PySpaceGeneratorNode>();
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_generate_design_space = std::move(f_generate_design_space);
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

}  // namespace meta_schedule
}  // namespace tvm
