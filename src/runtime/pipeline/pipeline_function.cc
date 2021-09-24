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
 * \param mod_configure Configure information that generate by export library function call.
 */
size_t PipelineFunction::PipelineInit(Array<Module> modules, const PipelineConfigure& pipeline_conf,
                                      const ModuleConfigure& mod_configure) {
  int outputNum = pipeline_conf.GetGlobalOutputNum();
  std::vector<Module> graphRuntimes = PipelineCreateGraphruntime(modules, mod_configure);
  return outputNum;
}
/*!
 * \brief There are two mode to create graph runtime list, first is to use modules that
 *  are the module list already created by caller, when modules is empty these information
 *  from mod_configure will get use to create graph runtime list.
 * \param modules List of graph runtime module.
 * \param mod_configure Configure information that generate by export library function call.
 */
std::vector<Module> PipelineFunction::PipelineCreateGraphruntime(
    Array<Module> modules, const ModuleConfigure& mod_configure) {
  const PackedFunc* graphRuntimeCreate = Registry::Get("tvm.graph_executor.create");
  std::vector<Module> ret;
  // if modules not empty just return in vector container
  if (!modules.empty()) {
    for (auto mod : modules) {
      ret.push_back(mod);
    }

    // if modules is empty, need to build the graph runtime from mod_conf
  } else {
    ret.resize(mod_configure.size());
    for (auto configure : mod_configure) {
      // load lib
      auto lib = Module::LoadFromFile(configure.second["lib_name"].c_str());

      // read json
      std::ifstream ifJson(configure.second["json_name"].c_str());
      if (ifJson.fail()) {
        throw std::runtime_error("json file not found!");
      }
      const std::string json((std::istreambuf_iterator<char>(ifJson)),
                             std::istreambuf_iterator<char>());

      // create graph runtime
      std::istringstream istr(configure.second["dev"]);
      std::string str;
      int deviceType = 1, deviceId = 0;
      while (getline(istr, str, ';')) {
        std::istringstream istrDev(str);
        std::string stemp;
        if (getline(istrDev, stemp)) {
          deviceType = stoi(stemp);
        }
        if (getline(istrDev, stemp)) {
          deviceId = stoi(stemp);
        }
      }
      Module graphModule = (*graphRuntimeCreate)(json, lib, deviceType, deviceId);

      // load parameter
      TVMByteArray params_arr;
      std::ifstream ifParam(configure.second["params"].c_str());
      if (ifParam.fail()) {
        throw std::runtime_error("params file not found!");
      }
      const std::string params((std::istreambuf_iterator<char>(ifParam)),
                               std::istreambuf_iterator<char>());
      params_arr.data = params.c_str();
      params_arr.size = params.length();
      auto load_params = graphModule.GetFunction("load_params");
      load_params(params_arr);

      // put into return vector
      ret[configure.first - 1] = graphModule;
    }
  }
  return ret;
}
}  // namespace runtime
}  // namespace tvm
