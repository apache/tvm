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
namespace tvm {
namespace runtime {
/*!
 * \brief Give frontends an access to packed functions.
 * \param name The name of the function.
 * \param sptr_to_self The pointer to the module node.
 * \return The corresponding packed function.
 */
PackedFunc PipelineExecutor::GetFunction(const String& name,
                                         const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_num_outputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->NumOutputs(); });
  } else if (name == "get_num_inputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->NumInputs(); });
  } else if (name == "get_input_pipeline_map") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        *rv = this->GetInputPipeplineMap(args[0].operator String());
      } else {
        LOG(FATAL) << "Function only support the input name value in the form of string";
      }
    });
  } else if (name == "get_params_group_pipeline_map") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        *rv = this->GetParamsGroupPipelineMap(args[0].operator String());
      } else {
        LOG(FATAL) << "Function only support the input name value in the form of string";
      }
    });
  } else if (name == "set_param") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0]) && String::CanConvertFrom(args[1])) {
        this->SetParam(args[0].operator String(), args[1].operator String(), args[2]);
      } else {
        LOG(FATAL) << "Function only support the parameter name and the key in the form of string";
      }
    });
  } else if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        this->SetInput(args[0].operator String(), args[1]);
      } else {
        LOG(FATAL) << "Function only support the input name value in the form of string";
      }
    });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        *rv = this->GetInput(args[0].operator String());
      } else {
        LOG(FATAL) << "Function only support the input name value in the form of string";
      }
    });
  } else if (name == "get_output") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetOutput(); });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->Run(); });
  } else if (name == "get_execute_count") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetExecutionCount(); });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
  }
}
/*!
 * brief Returns number of global inputs.
 */
int PipelineExecutor::NumInputs(void) { return input_connection_config_.GetInputNum(); }
/*!
 * \brief set input to the runtime module.
 * \param input_name The input name.
 * \param data_in The input data.
 */
void PipelineExecutor::SetInput(std::string input_name, DLTensor* data_in) {
  global_runtime_->SetPipelineInput(input_name, data_in);
}
/*!
 * \brief get input from the runtime module.
 * \param input_name The input name.
 * \return Return the input data for a specific input name.
 */
NDArray PipelineExecutor::GetInput(std::string input_name) {
  std::pair<int, int> indexs = this->GetInputIndex(input_name);
  if (indexs.first < 0 || indexs.first >= static_cast<int>(runtimes_.size())) {
    LOG(FATAL) << "input name " << input_name << " not found.";
  }
  return runtimes_[indexs.first]->GetInput(indexs.second);
}
/*!
 * \brief Getting a module index via a input parameters group name.
 * \param name The parameters group name.
 * \return int The module index.
 */
int PipelineExecutor::GetParamModuleIndex(const std::string& name) {
  return param_connection_config_[name];
}
/*!
 * \brief Using the global input name to get the index, and also get the input interface name
   of corresponding subgraph from the input connection configuration.
 * \param The global input name.
 * \return Returning the index and the input interface name of corresponding subgraph.
 */
Array<String> PipelineExecutor::GetInputPipeplineMap(std::string input_name) {
  std::pair<int, std::string> map = input_connection_config_[input_name];
  return {std::to_string(map.first), map.second};
}

/*!
 * \brief Return the module index for the parameters group name.
 * \param name The parameters group name.
 * \return int The module index.
 */
int PipelineExecutor::GetParamsGroupPipelineMap(const std::string& name) {
  return param_connection_config_[name];
}

/*!\brief Run the pipeline executor.*/
void PipelineExecutor::Run() { pipeline_scheduler_.PipelineRun(runtimes_); }
/*!
 * \brief return A list of global output data.
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
 * \brief Set a parameter into a graph module.
 * \param param_group_name The parameters group name.
 * \param param_key_name The parameter key name.
 * \param data_in The parameter data.
 */
void PipelineExecutor::SetParam(std::string param_group_name, std::string param_key_name,
                                DLTensor* data_in) {
  // Get the module index via the parameters group name.
  int module_index = this->GetParamModuleIndex(param_group_name);
  ICHECK(module_index >= 0 && module_index < static_cast<int>(runtimes_.size()))
      << "Parameter group name " << param_group_name << " does not exist.";
  auto runtime = runtimes_[module_index];
  // Get the parameter index via the param key name
  int index = runtime->GetInputIndex(param_key_name);
  ICHECK(index >= 0) << "Parameter name " << param_key_name << " does not exist in module "
                     << module_index;
  runtime->SetInput(index, data_in);
}
/*!
 * \brief Return the input index and module index for a given input name.
 * \param name The input name.
 * \return std::pair<int, int> A pair of module index and the input index.
 */
std::pair<int, int> PipelineExecutor::GetInputIndex(const std::string& name) {
  std::pair<int, std::string> index = input_connection_config_[name];
  auto gruntime = runtimes_[index.first];
  return std::make_pair(index.first, gruntime->GetInputIndex(index.second));
}
/*!
 * \brief Getting the count of running pipeline.
 */
int PipelineExecutor::GetExecutionCount() { return runtimes_.back()->GetExecutionCount(); }
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
  this->LoadConfig(&reader);
  ICHECK(!pipeline_config_.Empty()) << "The pipeline config information is empty.";
  num_outputs_ = pipeline_config_.GetGlobalOutputNum();
  // Initialize the pipeline function class used for pipeline thread pool management
  // and schedule etc. This function returns a list of runtime.
  global_runtime_ =
      pipeline_scheduler_.PipelineInit(modules, pipeline_config_, input_connection_config_);
  runtimes_ = global_runtime_->GetRuntimeList();
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
