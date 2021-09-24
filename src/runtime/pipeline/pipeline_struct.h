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
#ifndef TVM_RUNTIME_PIPELINE_PIPELINE_STRUCT_H_
#define TVM_RUNTIME_PIPELINE_PIPELINE_STRUCT_H_
#include <assert.h>
#include <dlpack/dlpack.h>
#include <dmlc/json.h>

#include <string>
#include <unordered_map>
#include <vector>
/*!
 * \brief Store the corresponding dependencies between the input of other modules
    and the current outputs.
 */
struct OutputBindings {
  /*! \brief All module interface binding with current output. */
  std::unordered_map<int, std::string> bindings;
  /*!
   * \brief If there is one global binding in bindings, then current output is
   *  global interface.
   * \return Whether this output interface is global output interface.
   */
  bool IsGlobalOutputNum() const {
    int outputNum = 0;
    for (auto binding : bindings) {
      /* output is global output when value is 0.
       */
      outputNum += (binding.first == 0);
    }
    /* If this output is a global output then there is only one such output in map.*/
    assert(outputNum <= 1);
    return outputNum == 1;
  }
  /*!
   * \brief Create module interface map from JSONReader.
   * \param reader Json reader.
   */
  void Load(dmlc::JSONReader* reader) {
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      std::string key;
      reader->BeginObject();
      std::string inputName;
      int mod_idx;
      while (reader->NextObjectItem(&key)) {
        if (key == "mod_idx") {
          reader->Read(&mod_idx);
        }
        if (key == "input_name") {
          reader->Read(&inputName);
        }
      }
      bindings[mod_idx] = inputName;
    }
  }
};
/*!
 * \brief Binding information of the outputs by each module.
 */
struct OutputMap {
  /*! \brief output and output binding map. */
  std::unordered_map<int, OutputBindings> output_binding_map;
  OutputMap& operator=(const OutputMap& output) {
    output_binding_map = output.output_binding_map;
    return *this;
  }
  /*! \brief Global output is the final outputs of pipeline, this function use to
   *   get how many global outputs are in this Outputmap
   *  \return Number of global outputs.
   */
  size_t GetGlobalOutputNum(void) const {
    size_t outputNum = 0;
    for (auto bindings : output_binding_map) {
      outputNum += bindings.second.IsGlobalOutputNum() ? 1 : 0;
    }
    return outputNum;
  }

  /*!
   * \brief Create output and output binding map from JSONReader.
   * \param reader Json reader.
   */
  void Load(dmlc::JSONReader* reader) {
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      std::string key;
      reader->BeginObject();
      int output_idx;
      OutputBindings binding;
      while (reader->NextObjectItem(&key)) {
        if (key == "output_idx") {
          reader->Read(&output_idx);
        }
        if (key == "dependent") {
          reader->Read(&binding);
        }
      }
      output_binding_map[output_idx] = binding;
    }
  }
};
/*!
 * \brief Binding or dependency information of each module output interface.
 */
struct PipelineConfigure {
  /*!  */
  std::unordered_map<int, OutputMap> config;
  OutputMap& operator[](const int key) { return config[key]; }
  /*!
   * \brief Get total global outputs number.
   * \return Global outputs number.
   */
  size_t GetGlobalOutputNum() const {
    size_t output_num = 0;
    for (auto mod_output : config) {
      output_num += mod_output.second.GetGlobalOutputNum();
    }
    return output_num;
  }
};
/*!
 * \brief Informations used to initialize the graph executor module, these
    information comming from export library function call.
 */
struct ModuleInformation {
  std::unordered_map<std::string, std::string> info;
  const std::string& operator[](const std::string& key) { return info[key]; }
  ModuleInformation& operator=(const std::unordered_map<std::string, std::string>& umap) {
    info = umap;
    return *this;
  }
};
/*! Module information of each module. */
typedef std::unordered_map<int, ModuleInformation> ModuleConfigure;
#endif  //  TVM_RUNTIME_PIPELINE_PIPELINE_STRUCT_H_
