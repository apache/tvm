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

void PipelineRuntime::Init(const Array<tvm::runtime::Module>& modules,
                           const std::string& pipeline_json) {
  return;
}

/* GetFunction can not be pure abstract function, implement an empty function for now.
 */
PackedFunc PipelineRuntime::GetFunction(const std::string& name,
                                        const ObjectPtr<Object>& sptr_to_self) {
  return nullptr;
}

Module PipelineRuntimeCreate(const Array<tvm::runtime::Module>& m,
                             const std::string& pipeline_json) {
  auto exec = make_object<PipelineRuntime>();
  exec->Init(m, pipeline_json);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.pipeline_executor.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = PipelineRuntimeCreate(args[0], args[1]);
});
}  // namespace runtime
}  // namespace tvm
