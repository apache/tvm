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
 * \brief All binding information of a output interface.
 */
struct OutputBindings {
  /*!\brief Output interface binding information, 'int' is the index of the bond module.
   * 'string' is the interface name of the bond module.
   */
  std::unordered_map<int, std::string> bindings;
  /*!
   * \brief If there is one global binding in bindings, then current output is
   *  global interface.
   * \return Whether this output interface is global output interface.
   */
  bool IsGlobalOutput() const {
    int num_output = 0;
    for (auto binding : bindings) {
      /* output is global output when value is 0.
       */
      num_output += (binding.first == 0);
    }
    /* If this output is a global output then there is only one such output in map.*/
    assert(num_output <= 1);
    return num_output == 1;
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
      std::string input_name;
      int mod_idx;
      while (reader->NextObjectItem(&key)) {
        if (key == "mod_idx") {
          reader->Read(&mod_idx);
        }
        if (key == "input_name") {
          reader->Read(&input_name);
        }
      }
      bindings[mod_idx] = input_name;
    }
  }
};
/*!
 * \brief The binding information of all outputs of a module.
 */
struct OutputMap {
  /*! \brief Output binding map, 'int' is output interface index.*/
  std::unordered_map<int, OutputBindings> output_binding_map;
  OutputMap& operator=(const OutputMap& output) {
    output_binding_map = output.output_binding_map;
    return *this;
  }

  /*!\brief Check that OutMap is empty.
   * \return True or False.
   */
  bool Empty() { return output_binding_map.empty(); }
  /*! \brief Global output is the final outputs of pipeline, this function use to
   *   get how many global outputs are in this Outputmap
   *  \return Number of global outputs.
   */
  size_t GetGlobalOutputNum(void) const {
    size_t num_output = 0;
    for (auto bindings : output_binding_map) {
      num_output += bindings.second.IsGlobalOutput() ? 1 : 0;
    }
    return num_output;
  }

  /*!
   * \brief Create output binding map from JSONReader.
   * \param reader Json reader.
   */
  void Load(dmlc::JSONReader* reader) {
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      std::string key;
      reader->BeginObject();
      int output_idx = -1;
      OutputBindings binding;
      while (reader->NextObjectItem(&key)) {
        if (key == "output_idx") {
          reader->Read(&output_idx);
        }
        if (key == "dependent") {
          reader->Read(&binding);
        }
      }
      assert(output_idx >= 0);
      output_binding_map[output_idx] = binding;
    }
  }
};
/*!
 * \brief Binding or dependency information of each module output interface.
 */
struct PipelineConfig {
  /*!\brief The module index is the key, this variable record all module pipeline configuration
   * information.
   */
  std::unordered_map<int, OutputMap> config;
  OutputMap& operator[](const int key) { return config[key]; }
  /*!
   * \brief Get the total global outputs number.
   * \return The global outputs number.
   */
  size_t GetGlobalOutputNum() const {
    size_t num_output = 0;
    for (auto mod_output : config) {
      num_output += mod_output.second.GetGlobalOutputNum();
    }
    return num_output;
  }
};
/*!
 * \brief The informations used to initialize the graph executor module, the information
 *  come from the export library function call.
 */
struct ModuleInformation {
  /*\brief The first string is the information type such as "lib_name",the second string is
   * the information detail such as "/src/lib1.so" for "lib_name".
   */
  std::unordered_map<std::string, std::string> info;
  const std::string& operator[](const std::string& key) { return info[key]; }
  ModuleInformation& operator=(const std::unordered_map<std::string, std::string>& umap) {
    info = umap;
    return *this;
  }
};
/*! The Module information of each module.The 'int' is module index. */
using ModuleConfig = std::unordered_map<int, ModuleInformation>;
#endif  //  TVM_RUNTIME_PIPELINE_PIPELINE_STRUCT_H_
