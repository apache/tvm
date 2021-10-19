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
 * \file pipeline_executor.cc
 */
#include "pipeline_executor.h"

#include <utility>
namespace tvm {
namespace runtime {
/*!
 * \brief Give frontends an access to packed functions.
 * \param name The name of the function.
 * \param sptr_to_self The pointer to the module node.
 * \return The corresponding packed function.
 */
PackedFunc PipelineExecutor::GetFunction(const std::string& name,
                                         const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_num_outputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->NumOutputs(); });
  } else if (name == "get_num_inputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->NumInputs(); });
  } else if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        this->SetInput(args[0].operator String(), args[1]);
      } else {
        LOG(FATAL) << "Function only support the input name value in the form of string";
      }
    });
  } else if (name == "set_param") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0]) && String::CanConvertFrom(args[1])) {
        this->SetParam(args[0].operator String(), args[1].operator String(), args[2]);
      } else {
        LOG(FATAL) << "Function only support the params name and keyin the form of string";
      }
    });
  } else if (name == "get_output") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetOutput(); });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        *rv = this->GetInput(args[0].operator String());
      } else {
        LOG(FATAL) << "Function only support the input name value in the form of string";
      }
    });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->Run(args[0]); });
  } else if (name == "stop") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->Stop(); });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc();
  }
  return nullptr;
}
/*!
 * \brief There are some input called pipeline global input that user need to use function
    "set_input" to set the data for it, this function return the number of such global input.
   \return Return the number of pipeline global input.
 */

int PipelineExecutor::NumInputs() const {
  // The number of inputs obtained from the input configuration.
  size_t config_inputs_num = input_connection_config.size(), ret = 0;
  // The number of inputs obtained from the graph runtime and pipeline configuration.
  size_t internal_inputs_num = pipeline_config_.GetInputOutputBindingNum();
  for (auto runtime : runtimes_) {
    ret += runtime->NumInputs();
  }
  // Use the summary of all backend runtime module input number to minus the internal inputs
  // number, then we will get the pipeline global input number
  ret -= internal_inputs_num;
  // Check whether these two numbers are equal.
  if (config_inputs_num != ret) {
    LOG(FATAL) << "The number of inputs from the configuration file is inconsistent!";
  }
  return ret;
}
/*!
 * \brief Return the input index and module index for a given input name.
 * \param name The input name.
 * \return std::pair<int, int> The module index and the input index.
 */
std::pair<int, int> PipelineExecutor::GetInputIndex(const std::string& name) {
  std::pair<int, std::string> index = input_connection_config[name];
  auto gruntime = runtimes_[index.first];
  return std::make_pair(index.first, gruntime->GetInputIndex(index.second));
}
/*!
 * \brief Return the module index for a given input param name.
 * \param name The params name.
 * \return int The module index.
 */
int PipelineExecutor::GetParamModuleIndex(const std::string& name) {
  return param_connection_config[name];
}
/*!
 * \brief set input to the graph module.
 * \param input_name The input name.
 * \param data_in The input data.
 */
void PipelineExecutor::SetInput(std::string input_name, DLTensor* data_in) {
  std::pair<int, int> indexs = this->GetInputIndex(input_name);
  runtimes_[indexs.first]->SetInput(indexs.second, data_in);
}

/*!
 * \brief get input from the graph module.
 * \param input_name The input name.
 * \return Return the input data for a specific input name.
 */
NDArray PipelineExecutor::GetInput(std::string input_name) {
  std::pair<int, int> indexs = this->GetInputIndex(input_name);
  return runtimes_[indexs.first]->GetInput(indexs.second);
}
/*!
 * \brief set param to a graph module.
 * \param input_name The input name.
 * \param data_in The input data.
 */
void PipelineExecutor::SetParam(std::string param_name, std::string param_key_name,
                                DLTensor* data_in) {
  // Get the module index from the param name.
  auto runtime = runtimes_[this->GetParamModuleIndex(param_name)];
  // Get the param index from the param key name
  int index = runtime->GetInputIndex(param_key_name);
  runtime->SetInput(index, data_in);
}
/*!
 * \brief Run the pipeline executor.
 * \param serialized_mode Whether run the pipeline executor in serialized mode.
 */
void PipelineExecutor::Run(bool serialized_mode) {
  pipeline_scheduler_.PipelineRun(runtimes_, pipeline_config_, serialized_mode);
}
/*!
 * \brief Stop the pipeline executor.
 */
void PipelineExecutor::Stop() { pipeline_scheduler_.PipelineStop(); }
/*!
 * \brief return A list of pipeline global output data.
 */
Array<NDArray> PipelineExecutor::GetOutput(void) { return pipeline_scheduler_.PipelineGetOutput(); }
/*!
 * \brief Use the mod_config information to create a graph runtime list.
 * \param mod_config The config information that generates by the export library function call.
 */
std::vector<Module> PipelineExecutor::CreateGraphModules(const ModuleConfig& mod_config) {
  const PackedFunc* graph_executor_create = Registry::Get("tvm.graph_executor.create");
  std::vector<Module> ret;
  ret.resize(mod_config.size());
  for (auto config : mod_config) {
    // Load library.
    auto lib = Module::LoadFromFile(config.second.lib_name.c_str());

    // Read json.
    std::ifstream ifJson(config.second.json_name.c_str());
    if (ifJson.fail()) {
      LOG(FATAL) << "json file not found: " << config.second.json_name;
    }
    const std::string json((std::istreambuf_iterator<char>(ifJson)),
                           std::istreambuf_iterator<char>());

    // Create a graph executor.
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
    const char* params_file_name = config.second.params_name.c_str();
    std::ifstream if_param(params_file_name);
    if (if_param.fail()) {
      LOG(FATAL) << "params file not found: " << params_file_name;
    }
    const std::string params((std::istreambuf_iterator<char>(if_param)),
                             std::istreambuf_iterator<char>());
    params_arr.data = params.c_str();
    params_arr.size = params.length();
    auto load_params = graph_module.GetFunction("load_params");
    load_params(params_arr);

    // Put a graph executor module into the vector.
    ret[config.first] = graph_module;
  }
  return ret;
}
/*!
 * \brief Initialize the pipeline executor with a list of modules to be pipelined
 *  and config in JSON format.
 * \param modules The module list used for building the pipeline.
 * \param pipeline_json The configuration of modules dependencies.
 */
void PipelineExecutor::Init(const std::vector<Module>& modules, const std::string& pipeline_json) {
  ICHECK(!modules.empty()) << "The graph executor module list is empty.";
  // Use JSONReader to load pipeline configuration.
  std::istringstream is(pipeline_json);
  dmlc::JSONReader reader(&is);
  ConfigPipelineExecution& pipeline_config = this->LoadConfigPipelineExecution(&reader);
  ICHECK(!pipeline_config.Empty()) << "The pipeline config information is empty.";
  num_outputs_ = pipeline_config.GetGlobalOutputNum();
  // Initialize the pipeline function class used for pipeline thread pool management
  // and schedule etc. This function returns a list of backend runtime.
  runtimes_ = pipeline_scheduler_.PipelineInit(modules, pipeline_config);
  return;
}

Module PipelineExecutorCreate(const Array<Module>& m, const std::string& pipeline_json) {
  ICHECK(!m.empty()) << "The module list is empty.";
  auto exec = make_object<PipelineExecutor>();
  std::vector<Module> graph_modules;
  for (auto mod : m) {
    graph_modules.push_back(mod);
  }
  exec->Init(graph_modules, pipeline_json);
  return Module(exec);
}

Module PipelineExecutorLoad(const std::string& load_json, const std::string& pipeline_json) {
  auto exec = make_object<PipelineExecutor>();
  std::istringstream is(load_json);
  dmlc::JSONReader reader(&is);
  ModuleConfig& mod_config = exec->LoadModuleConfig(&reader);
  ICHECK(!mod_config.empty()) << "The module config is empty.";
  std::vector<Module> modules = exec->CreateGraphModules(mod_config);
  exec->Init(modules, pipeline_json);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.pipeline_executor.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = PipelineExecutorCreate(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("tvm.pipeline_executor.load").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = PipelineExecutorLoad(args[0], args[1]);
});
}  // namespace runtime
}  // namespace tvm
