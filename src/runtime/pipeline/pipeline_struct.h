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
#include <utility>
#include <vector>
/*!
 * \brief All binding information of a output interface.
 */
class ConfigBindings {
 public:
  /*!\brief Whether this binding is bound to the PipelineExecutor output interface.*/
  bool IsGlobalOutput() const { return global_output_index_ > -1; }

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
      // Whether the output binding is global.
      bool global_binding = false;
      while (reader->NextObjectItem(&key)) {
        if (key == "mod_idx") {
          reader->Read(&mod_idx);
        } else if (key == "input_name") {
          reader->Read(&input_name);
        } else if (key == "global_output_index") {
          // There should be only one global binding.
          ICHECK(global_output_index_ < 0);
          reader->Read(&global_output_index_);
          // When the key value is 'global_output_index', it means that this output is bound to
          // a global interface.
          global_binding = true;
        } else {
          LOG(FATAL) << "do not support key " << key;
        }
      }
      // When this output is bound to a global interface, check if the global interface index
      // start from 0.
      if (global_binding) {
        ICHECK(global_output_index_ >= 0);
      } else {
        // When this output is bound to a graph executor module interface, check if the module
        // index start from 0.
        ICHECK(mod_idx >= 0);
        bindings_[mod_idx] = input_name;
      }
    }
  }

 private:
  /*!\brief Output interface binding information, 'int' is the index of the module that
   *  uses this output data as the input interface data, 'string' is the input interface name
   *  of the module.
   */
  std::unordered_map<int, std::string> bindings_;
  /*! The index value of the global interface to which the current output are bound.*/
  int global_output_index_ = std::numeric_limits<int>::min();
};
/*!
 * \brief The binding information of all outputs of a module.
 */
class ConfigOutputBindings {
 public:
  ConfigOutputBindings& operator=(const ConfigOutputBindings& output) {
    output_binding_map_ = output.GetOutBindings();
    return *this;
  }

  ConfigBindings& operator[](const int key) {
    ICHECK(output_binding_map_.find(key) != output_binding_map_.end());
    return output_binding_map_[key];
  }
  /*!brief Return the variable "output_binding_map_".*/
  std::unordered_map<int, ConfigBindings> GetOutBindings() const { return output_binding_map_; }
  /*!
   *\brief This function is used to verify whether ConfigOutputBindings is successfully loaded.
   *\return Return true to indicate that this class has not been successfully loaded.
   */
  bool Empty() { return output_binding_map_.empty(); }
  /*!
   * \brief The pipeline outputs is the final outputs of pipeline, this function is used to
   *  get how many pipeline outputs are in this Outputmap
   * \return Number of pipeline outputs.
   */
  size_t GetGlobalOutputNum(void) const {
    size_t num_output = 0;
    for (auto bindings : output_binding_map_) {
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
      ConfigBindings binding;
      while (reader->NextObjectItem(&key)) {
        if (key == "output_idx") {
          reader->Read(&output_idx);
        } else if (key == "dependencies") {
          reader->Read(&binding);
        } else {
          LOG(FATAL) << "do not support key " << key;
        }
      }
      ICHECK(output_idx >= 0);
      output_binding_map_[output_idx] = binding;
    }
  }

 private:
  /*!\brief The map of output binding, 'int' is the output interface index.*/
  std::unordered_map<int, ConfigBindings> output_binding_map_;
};

/*!
 * \brief The binding or dependency information of each module output interface.
 */
class ConfigPipelineExecution {
 public:
  /*
   *!\brief This function is used to verify whether config is loaded successfully.
   * \return Return "true" to indicate that this class has not been successfully loaded.
   */
  bool Empty() { return config_.empty(); }
  /*!
   * \brief Getting the number of global outputs.
   * \return The number of outputs in the entire pipeline.
   */
  size_t GetGlobalOutputNum() const {
    size_t num_output = 0;
    for (auto mod_output : config_) {
      num_output += mod_output.second.GetGlobalOutputNum();
    }
    return num_output;
  }
  /*!
   * \brief Create a pipeline config from JSONReader.
   * \param reader Json reader.
   */
  void Load(dmlc::JSONReader* reader) {
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      std::string key;
      reader->BeginObject();
      int mod_idx = -1;
      ConfigOutputBindings output;
      std::string dev;
      while (reader->NextObjectItem(&key)) {
        if (key == "mod_idx") {
          reader->Read(&mod_idx);
        } else if (key == "dev") {
          reader->Read(&dev);
        } else if (key == "output") {
          reader->Read(&output);
        } else {
          LOG(FATAL) << "do not support key " << key;
        }
      }
      ICHECK(mod_idx >= 0) << "Invalid mod_idx value " << mod_idx;
      // Check if the output is successfully read.
      ICHECK(!output.Empty()) << "Invalid output binding result.";
      // Build the mapping of mod_idx and "ConfigOutputBindings".
      config_[mod_idx] = output;
    }
  }

 private:
  /*
   *!\brief The key is the module index, this variable records all module pipeline configuration
   * information.
   */
  std::unordered_map<int, ConfigOutputBindings> config_;
};

struct InputConnectionConfig {
  /*!\brief The key("string") is the name of global module input interfaces. The value("pair")
   * includes the index of graph module and the name of a graph module input interface.
   */
  std::unordered_map<std::string, std::pair<int, std::string>> input_connection;
  std::pair<int, std::string> operator[](const std::string key) {
    if (input_connection.find(key) == input_connection.end()) {
      LOG(FATAL) << "Not find the key " << key;
    }
    return input_connection[key];
  }
  /*!
   * \brief Create a input connection config from JSONReader.
   * \param reader Json reader.
   */
  void Load(dmlc::JSONReader* reader) {
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      reader->BeginObject();
      std::string key;
      std::string global_interface_name;
      std::string module_interface_name;
      int mod_idx = -1;
      while (reader->NextObjectItem(&key)) {
        if (key == "global_interface_name") {
          reader->Read(&global_interface_name);
        } else if (key == "mod_idx") {
          reader->Read(&mod_idx);
        } else if (key == "module_interface_name") {
          reader->Read(&module_interface_name);
        } else {
          LOG(FATAL) << "do not support key " << key;
        }
      }
      ICHECK(mod_idx >= 0) << "Invalid mod_idx value " << mod_idx;
      ICHECK(!global_interface_name.empty()) << "Invalid global interface name value";
      ICHECK(!module_interface_name.empty()) << "Invalid module interface name value";
      input_connection[global_interface_name] = make_pair(mod_idx, module_interface_name);
    }
  }
};

/*!
 * \brief The information used to initialize the graph executor module, the information
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
