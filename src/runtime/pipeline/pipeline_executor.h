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

#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pipeline_function.h"
namespace tvm {
namespace runtime {
/*!
 * \brief pipeline executor.
 *  This executor class use the module list and dependency configuration of modules as
 *  the parameters and executes these modules on heterogeneous targets in a pipeline
 *  parallel manner to improve throughput.
 *
 *  This executor can be accessed by various language via TVM runtime PackedFunc API.
 */
class TVM_DLL PipelineExecutor : public ModuleNode {
 public:
  /*!
   * \Return the type key of the executor.
   */
  const char* type_key() const final { return "PipelineExecutor"; }
  /*!
   * \brief Initialize the pipeline executor with module array and json text.
   * \param modules The module list used for building pipeline.
   * \param pipeline_json The configuration of modules dependencies.
   */
  void Init(const Array<Module>& modules, const std::string& pipeline_json);
  /*!
   * \brief Give frontends an access to packed functions.
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding packed function.
   */
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  /*!
   * \brief Get the number of outputs.
   *
   * \return The number of outputs.
   */
  int NumOutputs() const { return output_number_; }

 private:
  /*!\brief The class used to execute pipeline logic*/
  PipelineFunction pipeline_function_;
  /*!\brief Dependency information of each graph runtime module of pipe line.*/
  PipelineConfigure pipeline_configure_;
  /*!\brief Module information that can get use to create graph runtime.*/
  ModuleConfigure mod_configure_;
  /*!\birief How many outputs are in this pipeline executor.*/
  size_t output_number_ = 0;
  /*!\brief Json loader.*/
  void Load(dmlc::JSONReader* reader) {
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      std::string key;
      reader->BeginObject();
      int mod_idx = 0;
      std::string libName;
      std::string jsonName;
      std::string paramsName;
      std::string dev;
      OutputMap output;
      while (reader->NextObjectItem(&key)) {
        if (key == "mod_idx") {
          reader->Read(&mod_idx);
        }
        if (key == "lib_name") {
          reader->Read(&libName);
        }

        if (key == "json_name") {
          reader->Read(&jsonName);
        }

        if (key == "params_name") {
          reader->Read(&paramsName);
        }

        if (key == "dev") {
          reader->Read(&dev);
        }

        if (key == "output") {
          reader->Read(&output);
        }
      }
      pipeline_configure_[mod_idx] = output;
      mod_configure_[mod_idx] = {
          {"lib_name", libName}, {"json_name", jsonName}, {"params", paramsName}, {"dev", dev}};
    }
  }
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_PIPELINE_PIPELINE_EXECUTOR_H_
