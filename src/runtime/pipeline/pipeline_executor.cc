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

/*!
 * \file pipeline_executor.cc
 */
#include "pipeline_executor.h"
namespace tvm {
namespace runtime {
/*!
 * \brief Give frontends an access to packed functions.
 * \param name The name of the function.
 * \param sptr_to_self The pointer to the module node.
 * \return The corresponding packed function.
 */
PackedFunc PipelineExecutor::GetFunction(const std::string& name,
                                         const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_num_outputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->NumOutputs(); });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc();
  }
  return nullptr;
}
/*!
 * \brief Initialize the pipeline executor with a list of modules to be pipelined
 *  and config in JSON format.
 * \param modules The module list used for building pipeline.
 * \param pipeline_json The configuration of modules dependencies.
 */
void PipelineExecutor::Init(const Array<Module>& modules, const std::string& pipeline_json) {
  // Use JSONReader to load pipeline configuration from file.
  std::istringstream is(pipeline_json);
  dmlc::JSONReader reader(&is);
  this->Load(&reader);
  // Initialize the pipeline function class used for pipeline thread pool management
  // and schedule etc. This function returns the number of output.
  num_outputs_ = pipeline_function_.PipelineInit(modules, pipeline_config_, mod_config_);
  return;
}

Module PipelineExecutorCreate(const Array<Module>& m, const std::string& pipeline_json) {
  auto exec = make_object<PipelineExecutor>();
  exec->Init(m, pipeline_json);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.pipeline_executor.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = PipelineExecutorCreate(args[0], args[1]);
});
}  // namespace runtime
}  // namespace tvm
