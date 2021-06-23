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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "pipeline_function.h"

using namespace std;
namespace tvm {
namespace runtime {

/*!
 * \brief pipeline runtime.
 *
 *  This runtime can be acccesibly in various language via
 *  TVM runtime PackedFunc API.
 */
class TVM_DLL SubGraphRuntime : public ModuleNode {
 public:
  SubGraphRuntime() { input_int_map_ = make_shared<MOD_DLDATA_MAP>(); }
  ~SubGraphRuntime() {
    /* stop pipeline threads and release data in deconstructor.
     */
    Stop();
  }
  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final { return "SubGraphRuntime"; }
  void Run();
  void Stop();

  /*!
   * \brief Initialize the graph executor with graph and context.
   * \param graph_json The execution graph.
   * \param module The module containing the compiled functions for the host
   *  processor.
   * \param ctxs The context of the host and devices where graph nodes will be
   *  executed on.
   * \param lookup_linked_param_func If given, a PackedFunc invoked to lookup linked parameters
   *  by storage_id. If not given, linked parameters are looked-up using an internal implementation,
   *  which is not compatible with RPCModules.
   */
  void Init(const Array<tvm::runtime::Module>& modules, const std::string& pipeline_json);

  /*!
   * \brief set index-th input to the graph.
   * \param index The input index.
   * \param data_in The input data.
   */
  void SetInput(int index, DLTensor* data_in, int mod_idx);

  /*!
   * \brief get index-th input.
   * \param index The input index.
   * \return The input data.
   */
  NDArray GetInput(int index, int mIndx) const;

  /*!
   * \brief get input index-th by name.
   * \param input name.
   * \return The input index.
   */
  int GetInputIndex(const string& name, int mIndx) const;
  /*!
   * \brief Get the number of outputs
   *
   * \return The number of outputs from graph.
   */
  int NumOutputs() const;
  /*!
   * \brief Get the number of inputs
   *
   * \return The number of inputs to the graph.
   */
  int NumInputs() const;
  /*!
   * \brief Return NDArray Array for all output.
   *
   * \param syncPoll Syncholization poll mode or ASyncholization.
   * \return NDArray Array for all output.
   */
  Array<NDArray> GetOutput(bool syncPoll = true);

  void Load(dmlc::JSONReader* reader) {
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      std::string key;
      reader->BeginObject();
      int mod_indx = 0;
      std::string libName;
      std::string jsonName;
      std::string paramsName;
      std::string dev;
      unordered_map<int, unordered_map<int, string>> output;
      unordered_map<int, unordered_map<string, string>> lib;
      while (reader->NextObjectItem(&key)) {
        if (key == "mod_indx") {
          reader->Read(&mod_indx);
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
          reader->BeginArray();
          while (reader->NextArrayItem()) {
            int output_indx = -1;
            unordered_map<int, string> depend;
            reader->BeginObject();
            while (reader->NextObjectItem(&key)) {
              if (key == "output_indx") {
                reader->Read(&output_indx);
              }
              if (key == "dependent") {
                reader->BeginArray();
                int dep_mod_indx = -1;
                string inputName;
                while (reader->NextArrayItem()) {
                  reader->BeginObject();
                  while (reader->NextObjectItem(&key)) {
                    if (key == "mod_indx") {
                      reader->Read(&dep_mod_indx);
                    }
                    if (key == "input_name") {
                      reader->Read(&inputName);
                    }
                  }
                  if (dep_mod_indx >= 0) {
                    depend[dep_mod_indx] = inputName;
                  }
                }
              }
            }

            if (output_indx >= 0) {
              output[output_indx] = depend;
            }
          }
        }
      }
      if (mod_indx >= 0) {
        pipeline_conf_[mod_indx] = output;
        mod_conf_[mod_indx] = {
            {"lib_name", libName}, {"json_name", jsonName}, {"params", paramsName}, {"dev", dev}};
      }
    }
  }

 protected:
  vector<NDArray> output_entry_;
  PIPELINE_CONF pipeline_conf_;
  MOD_CONF mod_conf_;
  vector<shared_ptr<RuntimeItem>> runtimes_;
  MOD_DLDATA_MAP_PTR input_int_map_;
  size_t outpuNumber_ = 0;
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_PIPELINE_PIPELINE_EXECUTOR_H_
