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
#include "pipeline_function.h"

#include <utility>
#include <vector>
namespace tvm {
namespace runtime {
/*!
 * \brief Initialize the pipeline.
 * \param modules The list of graph executor module.
 * \param pipeline_conf The Dependency information of each graph executor module.
 * \param mod_config The config information that generated by the export library function call.
 */
size_t PipelineFunction::PipelineInit(Array<Module> modules, const PipelineConfig& pipeline_config,
                                      const ModuleConfig& mod_config) {
  int num_output = pipeline_config.GetGlobalOutputNum();
  // If 'modules' is not empty just return in vector container
  if (!modules.empty()) {
    for (auto mod : modules) {
      graph_executors_.push_back(mod);
    }
  } else {
    // If 'modules' is empty, need to build the graph exectuor from mod_config.
    graph_executors_ = PipelineCreateGraphExecutors(mod_config);
  }
  return num_output;
}
/*!
 * \brief Use the mod_config information to create a graph runtime list.
 * \param mod_config The config information that generate by export library function call.
 */
std::vector<Module> PipelineFunction::PipelineCreateGraphExecutors(const ModuleConfig& mod_config) {
  const PackedFunc* graph_executor_create = Registry::Get("tvm.graph_executor.create");
  std::vector<Module> ret;
  ret.resize(mod_config.size());
  for (auto config : mod_config) {
    // Load library.
    auto lib = Module::LoadFromFile(config.second.lib_name.c_str());

    // Read json.
    std::ifstream ifJson(config.second.json_name.c_str());
    if (ifJson.fail()) {
      LOG(FATAL) << "json file not found!";
    }
    const std::string json((std::istreambuf_iterator<char>(ifJson)),
                           std::istreambuf_iterator<char>());

    // Create graph executor.
    std::istringstream istr(config.second.dev);
    std::string str;
    int device_type = 1, device_id = 0;
    while (getline(istr, str, ';')) {
      std::istringstream istr_dev(str);
      std::string str_temp;
      if (getline(istr_dev, str_temp)) {
        device_type = stoi(str_temp);
      }
      if (getline(istr_dev, str_temp)) {
        device_id = stoi(str_temp);
      }
    }
    Module graph_module = (*graph_executor_create)(json, lib, device_type, device_id);

    // Load parameters.
    TVMByteArray params_arr;
    std::ifstream if_param(config.second.params_name.c_str());
    if (if_param.fail()) {
      LOG(FATAL) << "params file not found!";
    }
    const std::string params((std::istreambuf_iterator<char>(if_param)),
                             std::istreambuf_iterator<char>());
    params_arr.data = params.c_str();
    params_arr.size = params.length();
    auto load_params = graph_module.GetFunction("load_params");
    load_params(params_arr);

    // Put a graph executor module into the vector. because 'config.first' start from 1, use
    // 'config.first - 1' here to get the correct index value of module in the vector.
    ret[config.first - 1] = graph_module;
  }
  return ret;
}
}  // namespace runtime
}  // namespace tvm
