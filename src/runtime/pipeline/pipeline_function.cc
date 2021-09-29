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
 * \brief Initialize pipeline.
 * \param modules  List of graph runtime module.
 * \param pipeline_conf Dependency relation of each graph runtime module.
 * \param mod_config Config information that generate by export library function call.
 */
size_t PipelineFunction::PipelineInit(Array<Module> modules, const PipelineConfig& pipeline_config,
                                      const ModuleConfig& mod_config) {
  int num_output = pipeline_config.GetGlobalOutputNum();
  // if modules not empty just return in vector container
  if (!modules.empty()) {
    for (auto mod : modules) {
      graph_executors_.push_back(mod);
    }
  } else {
    // if modules is empty, need to build the graph runtime from mod_config.
    graph_executors_ = PipelineCreateGraphExecutors(mod_config);
  }
  return num_output;
}
/*!
 * \brief Use mod_config information to create a graph runtime list.
 * \param mod_configure Config information that generate by export library function call.
 */
std::vector<Module> PipelineFunction::PipelineCreateGraphExecutors(const ModuleConfig& mod_config) {
  const PackedFunc* graph_executor_create = Registry::Get("tvm.graph_executor.create");
  std::vector<Module> ret;
  ret.resize(mod_config.size());
  for (auto config : mod_config) {
    // load lib
    auto lib = Module::LoadFromFile(config.second["lib_name"].c_str());

    // read json
    std::ifstream ifJson(config.second["json_name"].c_str());
    if (ifJson.fail()) {
      throw std::runtime_error("json file not found!");
    }
    const std::string json((std::istreambuf_iterator<char>(ifJson)),
                           std::istreambuf_iterator<char>());

    // create graph runtime
    std::istringstream istr(config.second["dev"]);
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

    // load parameter
    TVMByteArray params_arr;
    std::ifstream if_param(config.second["params"].c_str());
    if (if_param.fail()) {
      throw std::runtime_error("params file not found!");
    }
    const std::string params((std::istreambuf_iterator<char>(if_param)),
                             std::istreambuf_iterator<char>());
    params_arr.data = params.c_str();
    params_arr.size = params.length();
    auto load_params = graph_module.GetFunction("load_params");
    load_params(params_arr);

    // put into return vector
    ret[config.first - 1] = graph_module;
  }
  return ret;
}
}  // namespace runtime
}  // namespace tvm
