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
 * \brief pipeline executor
 * \file pipeline_executor.h
 */
#ifndef TVM_RUNTIME_PIPELINE_PIPELINE_EXECUTOR_H_
#define TVM_RUNTIME_PIPELINE_PIPELINE_EXECUTOR_H_
#include <tvm/runtime/registry.h>

#include <string>
namespace tvm {
namespace runtime {
/*!
 * \brief pipeline executor.
 *  This executor take Module list and Dependency relations of Modules as
 *  inputs, then execute modules on heterogeneous hardware target in
 *  pipeline parallism mode to improve processing throughput. 
 *
 *  This executor can be acccesibly in various language via
 *  TVM runtime PackedFunc API.
 */
class TVM_DLL PipelineRuntime : public ModuleNode {
 public:
  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final { return "PipelineRuntime"; }
  /*!
   * \brief Initialize the pipeline executor with module Array and json text.
   * \param modules The module list that use for pipeline build.
   * \param pipeline_json The configuration for module dependent in pipeline.
   */
  void Init(const Array<tvm::runtime::Module>& modules, const std::string& pipeline_json);
  /*!
   * \brief Give frontends an access to packed functions.
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding packed functions.
   */
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_PIPELINE_PIPELINE_EXECUTOR_H_
