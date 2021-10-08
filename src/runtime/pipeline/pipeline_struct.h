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

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#define PIPELINE_EXECUTOR_INDEX -1
/*!
 * \brief All binding information of a output interface.
 */
struct OutputBindings {
  /*!\brief Output interface binding information, 'int' is the index of the module that
   *  uses this output data as the input interface data, 'string' is the input interface name
   *  of the module.
   */
  std::unordered_map<int, std::string> bindings;
  /*!
   * \brief If there is one PipelineExecutor binding in bindings, then the current output is
   *  PipelineExecutor interface.
   * \return Whether this output interface is PipelineExecutor output interface.
   */
  bool IsGlobalOutput() const {
    int num_output = 0;
    for (auto binding : bindings) {
      /* The output is a PipelineExecutor output when the index value
       * equal PIPELINE_EXECUTOR_INDEX.
       */
      num_output += (binding.first == PIPELINE_EXECUTOR_INDEX);
    }
    /* If this output is a global output then there is only one such output in map.*/
    ICHECK(num_output <= 1);
    return num_output == 1;
  }
  /*!
   * \brief Create a module interface map from JSONReader.
   * \param reader JSON reader.
   */
  void Load(dmlc::JSONReader* reader) {
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      std::string key;
      reader->BeginObject();
      std::string input_name;
      int mod_idx = std::numeric_limits<int>::min();
      while (reader->NextObjectItem(&key)) {
        if (key == "mod_idx") {
          reader->Read(&mod_idx);
        }
        if (key == "input_name") {
          reader->Read(&input_name);
        }
      }
      // In this level 'Load' that reading the output binding , the module can be
      // a 'PipelineExecutor', hence the value of 'mod_idx' should start from
      // PIPELINE_EXECUTOR_INDEX.
      ICHECK(mod_idx >= PIPELINE_EXECUTOR_INDEX);
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

  /*!\brief This function is used to verify whether OutputMap is loaded successfully.
   * \return Return true to indicate that this class has not been successfully loaded.
   */
  bool Empty() { return output_binding_map.empty(); }
  /*! \brief The pipeline outputs is the final outputs of pipeline, this function is used to
   *   get how many pipeline outputs are in this Outputmap
   *  \return Number of pipeline outputs.
   */
  size_t GetGlobalOutputNum(void) const {
    size_t num_output = 0;
    for (auto bindings : output_binding_map) {
      num_output += bindings.second.IsGlobalOutput() ? 1 : 0;
    }
    return num_output;
  }

  /*!
   * \brief Create a output binding map from JSONReader.
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
      ICHECK(output_idx >= 0);
      output_binding_map[output_idx] = binding;
    }
  }
};
/*!
 * \brief The binding or dependency information of each module output interface.
 */
struct PipelineConfig {
  /*!\brief The module index is the key, this variable records all module pipeline configuration
   * information.
   */
  std::unordered_map<int, OutputMap> config;
  OutputMap& operator[](int key) {
    ICHECK(config.find(key) != config.end());
    return config[key];
  }

  void Insert(int key, const OutputMap& map) { config[key] = map; }

  /*!\brief This function is used to verify whether config is loaded successfully.
   * \return Return true to indicate that this class has not been successfully loaded.
   */
  bool Empty() { return config.empty(); }

  /*!
   * \brief Get the number of global outputs that is the outputs of entire pipeline.
   * \return How much output does the entire pipeline have.
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
struct GraphModuleLoadInfo {
  GraphModuleLoadInfo(const std::string& lib, const std::string& json, const std::string& params,
                      const std::string& device)
      : lib_name(lib), json_name(json), params_name(params), dev(device) {}
  GraphModuleLoadInfo() { ; }
  std::string lib_name;
  std::string json_name;
  std::string params_name;
  std::string dev;
};
/*! The Module information of each module.The 'int' is module index. */
using ModuleConfig = std::unordered_map<int, GraphModuleLoadInfo>;
#endif  //  TVM_RUNTIME_PIPELINE_PIPELINE_STRUCT_H_
