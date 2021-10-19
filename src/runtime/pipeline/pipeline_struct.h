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
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace tvm {
namespace runtime {
#define GLOBAL_MODULE_INDEX -1
/*!
 *\brief The mapping of a module output and a global output in the graph module.
 */
struct GlobalOutputPair {
  int mod_output_idx;
  int global_output_idx;
  GlobalOutputPair(const int idx, const int gidx) : mod_output_idx(idx), global_output_idx(gidx) {}
  GlobalOutputPair() {}
};

/*!
 *\brief Use the module index and the output index to specify a module output.
 */
struct ModuleOutputPair {
  int mod_idx;
  int output_idx;
  ModuleOutputPair(const int midx, const int idx) : mod_idx(midx), output_idx(idx) {}
  ModuleOutputPair() {}
};
/*!
 * \brief All binding information of a output interface.
 */
class ConfigBindings {
 private:
  /*!\brief Output interface binding information, 'int' is the index of the module that
   *  uses this output data as the input interface data, 'string' is the input interface name
   *  of the module.
   */
  std::unordered_map<int, std::string> bindings;
  /*! The index value of the global interface to which the current output are bound.*/
  int global_output_index = std::numeric_limits<int>::min();

 public:
  /*!
   *\brief Return the memeber variable "bindings".
   */
  std::unordered_map<int, std::string>& Get() { return bindings; }
  /*!\brief Get the value of global outpu index.*/
  int GetGlobalOutputIndex() const { return global_output_index; }
  /*!\brief Whether this binding is bound to the PipelineExecutor output interface.*/
  bool IsGlobalOutput() const { return global_output_index >= 0; }

  /*!
   *\brief The number of bindings of input and output. one input only can bind with one
   * specific output, hence this number also is the number that how many module input data
   * source is internal moudle output.
   * return The number of binding in this module.
   */
  size_t GetInputOutputBindingNum(void) {
    size_t ret = 0;
    for (auto connection : bindings) {
      // Filter out the global output.
      if (connection.first == GLOBAL_MODULE_INDEX) continue;
      ret++;
    }
    return ret;
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
      // Whether the output binding is global.
      bool global_binding = false;
      while (reader->NextObjectItem(&key)) {
        if (key == "mod_idx") {
          reader->Read(&mod_idx);
        } else if (key == "input_name") {
          reader->Read(&input_name);
        } else if (key == "global_output_index") {
          // There should be only one global binding.
          ICHECK(global_output_index < 0);
          reader->Read(&global_output_index);
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
        ICHECK(global_output_index >= 0);
      } else {
        // When this output is bound to a graph executor module interface, check if the module
        // index start from 0.
        ICHECK(mod_idx >= 0);
        bindings[mod_idx] = input_name;
      }
    }
  }
};
/*!
 * \brief The binding information of all outputs of a module.
 */
class ConfigOutputBindings {
 private:
  /*! \brief Output binding map, 'int' is output interface index.*/
  std::unordered_map<int, ConfigBindings> output_binding_map;

 public:
  ConfigOutputBindings& operator=(const ConfigOutputBindings& output) {
    output_binding_map = output.output_binding_map;
    return *this;
  }

  ConfigBindings& operator[](const int key) {
    ICHECK(output_binding_map.find(key) != output_binding_map.end());
    return output_binding_map[key];
  }
  /*!
   *\brief Check if there is a output with the specify index in this map.
   */
  bool FindOutputInMap(int output_idx) {
    return output_binding_map.find(output_idx) != output_binding_map.end();
  }
  /*!
   *\brief This function is used to verify whether ConfigOutputBindings is successfully loaded.
   *\return Return true to indicate that this class has not been successfully loaded.
   */
  bool Empty() { return output_binding_map.empty(); }
  /*!
   * \brief The pipeline outputs is the final outputs of pipeline, this function is used to
   *  get how many pipeline outputs are in this Outputmap
   * \return Number of pipeline outputs.
   */
  size_t GetGlobalOutputNum(void) const {
    size_t num_output = 0;
    for (auto bindings : output_binding_map) {
      num_output += bindings.second.IsGlobalOutput() ? 1 : 0;
    }
    return num_output;
  }
  /*!
   *\brief Get the mapping of all global outputs and module outputs in this module.
   *\return A list of "GlobalOutputPair".
   */
  std::vector<GlobalOutputPair> GetGlobalConfigOutputBindings(void) {
    std::vector<GlobalOutputPair> ret;
    for (auto bindings : output_binding_map) {
      if (bindings.second.IsGlobalOutput()) {
        ret.push_back(GlobalOutputPair(bindings.first, bindings.second.GetGlobalOutputIndex()));
      }
    }
    return ret;
  }
  /*!
   *\brief How many inputs are binding with a backend module output in this module.
   */
  size_t GetInputOutputBindingNum() {
    size_t ret = 0;
    for (auto x : output_binding_map) {
      ret += x.second.GetInputOutputBindingNum();
    }
    return ret;
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
      output_binding_map[output_idx] = binding;
    }
  }
};

/*!
 * \brief A map of the global module input interfaces and the graph modudles input interfaces.
 */
struct InputConnectionConfig {
  /*!\brief The key is the name of global module input interfaces. the value is the pair of
   * the index of a graph module and the name of a graph module input interface.
   */
  std::unordered_map<std::string, std::pair<int, std::string>> input_connection;
  bool Empty() { return input_connection.empty(); }
  std::pair<int, std::string> operator[](const std::string key) {
    if (input_connection.find(key) == input_connection.end()) {
      LOG(FATAL) << "Not find the key " << key;
    }
    return input_connection[key];
  }

  size_t size() const { return input_connection.size(); }
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
 * \brief A map of the global module param interfaces and the graph modudles param.
 */
struct ParamConnectionConfig {
  /*!\brief The key is the name of global module param interfaces. the value is the
   * index of a graph module.
   */
  std::unordered_map<std::string, int> param_connection;
  bool Empty() { return param_connection.empty(); }
  int operator[](const std::string key) {
    if (param_connection.find(key) == param_connection.end()) {
      LOG(FATAL) << "do not support key " << key;
    }
    return param_connection[key];
  }
  /*!
   * \brief Create a param connection config from JSONReader.
   * \param reader Json reader.
   */
  void Load(dmlc::JSONReader* reader) {
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      reader->BeginObject();
      std::string key;
      std::string global_param_name;
      int mod_idx = -1;
      while (reader->NextObjectItem(&key)) {
        if (key == "global_param_name") {
          reader->Read(&global_param_name);
        } else if (key == "mod_idx") {
          reader->Read(&mod_idx);
        } else {
          LOG(FATAL) << "do not support key " << key;
        }
      }
      ICHECK(mod_idx >= 0) << "Invalid mod_idx value " << mod_idx;
      ICHECK(!global_param_name.empty()) << "Invalid global param name value";
      param_connection[global_param_name] = mod_idx;
    }
  }
};
/*!
 * \brief The binding or dependency information of each module output interface.
 */
class ConfigPipelineExecution {
 private:
  /*
   *!\brief The key is the module index, this variable records all module pipeline configuration
   * information.
   */
  std::unordered_map<int, ConfigOutputBindings> config;
  /*
   *\brief The key is the global output index, this variable records the mapping of global output
   * and the module output.
   */
  std::unordered_map<int, ModuleOutputPair> global_output_map;
  /*
   *\brief The number of binding of module outputs and inputs.
   */
  size_t module_input_output_binding_total_num;

 public:
  ConfigOutputBindings& operator[](int key) {
    ICHECK(config.find(key) != config.end());
    return config[key];
  }
  /*!
   *\brief Check if the module index existing in the "config".
   */
  bool FindModuleInConfig(int mod_idx) { return config.find(mod_idx) != config.end(); }
  /*!
   *\brief Build the mapping of key and "ConfigOutputBindings", key is module index.
   */
  void Insert(int key, const ConfigOutputBindings& map) { config[key] = map; }

  /*
   *!\brief This function is used to verify whether config is loaded successfully.
   * \return Return true to indicate that this class has not been successfully loaded.
   */
  bool Empty() { return config.empty(); }
  /*!
   * \brief Get the number of global outputs.
   * \return The number of outputs the entire pipeline has.
   */
  size_t GetGlobalOutputNum() const {
    // The number of pipeline outputs is the size of "global_output_map";
    return global_output_map.size();
  }
  /*
   *!\brief Get the map of global outputs and module outputs.
   */
  std::unordered_map<int, ModuleOutputPair>& GetGlobalConfigOutputBindings(void) {
    return global_output_map;
  }
  /*
   *!\brief Get the number of module output and module input bindings.
   */
  size_t GetInputOutputBindingNum() const { return module_input_output_binding_total_num; }
  /*
   *!\brief Parse the config to construct data struct using in pipeline execution.
   */
  void ParseConfiguration(const std::unordered_map<int, ConfigOutputBindings>& config) {
    if (config.empty()) {
      LOG(FATAL) << "The Configuration loading not finish yet.";
    }
    module_input_output_binding_total_num = 0;
    for (auto mod_output : config) {
      // Get the numbers of binding of input and output.
      module_input_output_binding_total_num += mod_output.second.GetInputOutputBindingNum();
      // Use global output index as key to create a mapping of global index and module output.
      const std::vector<GlobalOutputPair>& global_output =
          mod_output.second.GetGlobalConfigOutputBindings();

      for (auto output : global_output) {
        global_output_map[output.global_output_idx] =
            ModuleOutputPair(mod_output.first, output.mod_output_idx);
      }
    }
    return;
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
      Insert(mod_idx, output);
    }
    // Call this function after "config" loading finished.
    ParseConfiguration(config);
  }
};
/*
 *\brief Runtime of backend.
 */
class BackendRuntime {
 private:
  /*\brief The index of runtime indicate the position in the pipeline.*/
  int runtime_idx;
  /*\brief The Runtime module of a backedn graph executor.*/
  Module module;
  /*!
   *\brief To transfer data between two different backends, we need a local
   * tensor variable as a medium. This variable is a mapping of input data and local
   * data.
   */
  std::unordered_map<DLTensor*, DLTensor*> input_tensor_local_copy;
  /*!\brief The packed functions.*/
  tvm::runtime::PackedFunc run;
  tvm::runtime::PackedFunc set_input;
  tvm::runtime::PackedFunc get_input;
  tvm::runtime::PackedFunc get_output;
  tvm::runtime::PackedFunc get_num_output;
  tvm::runtime::PackedFunc get_num_inputs;
  tvm::runtime::PackedFunc get_input_index;
  /*!\brief The new DLTensor have same shape, data type with a existing DLTensor.*/
  DLTensor* CreateFromDLTensor(const DLTensor* from) {
    DLTensor* ret = NULL;
    TVMArrayAlloc(from->shape, from->ndim, from->dtype.code, from->dtype.bits, from->dtype.lanes,
                  kDLCPU, 0, &ret);
    return ret;
  }
  /*!\brief The new NDArray have same shape, data type with an existing DLTensor.*/
  NDArray CreateNDArrayFromDLTensor(const DLTensor* from) {
    std::vector<int64_t> shape;
    for (int i = 0; i < from->ndim; i++) {
      shape.push_back(from->shape[i]);
    }
    auto ndarray = NDArray::Empty(shape, from->dtype, from->device);
    ndarray.CreateView(shape, from->dtype);
    return ndarray;
  }
  /*
   *\brief Copy data from a DLTensor to another DLTensor.
   */
  void CopyFromTo(DLTensor* from, DLTensor* to) {
    // If the source device and target device is not same, we use a local DLTensor
    // as a medium to do the cross device copy work.
    if (!(from->device.device_type == to->device.device_type ||
          from->device.device_type == kDLCPU || to->device.device_type == kDLCPU)) {
      DLTensor* dltensor_local = nullptr;
      if (input_tensor_local_copy.find(to) == input_tensor_local_copy.end()) {
        dltensor_local = CreateFromDLTensor(from);
        input_tensor_local_copy[to] = dltensor_local;
      } else {
        dltensor_local = input_tensor_local_copy[to];
      }
      TVMArrayCopyFromTo(from, dltensor_local, nullptr);
      from = dltensor_local;
    }

    TVMArrayCopyFromTo(from, to, nullptr);
  }

 public:
  BackendRuntime(Module mod, int mod_idx) {
    module = mod;
    runtime_idx = mod_idx;
    get_input_index = module.GetFunction("get_input_index");
    get_num_output = module.GetFunction("get_num_outputs");
    get_num_inputs = module.GetFunction("get_num_inputs");
    set_input = module.GetFunction("set_input");
    get_input = module.GetFunction("get_input");
    get_output = module.GetFunction("get_output");
    run = module.GetFunction("run");
  }
  BackendRuntime(void) {}
  ~BackendRuntime() {
    for (auto data : input_tensor_local_copy) {
      TVMArrayFree(data.second);
    }
  }
  /*!\brief Create a new NDArray which have same shape, data type with a module output. */
  NDArray CreateFromOutput(int idx) {
    NDArray data = get_output(idx);
    return CreateNDArrayFromDLTensor(const_cast<DLTensor*>(data.operator->()));
  }
  /*!\brief Return the moudle index.*/
  int GetModuleIndex() { return runtime_idx; }
  /*!\brief Return the number of output*/
  int NumOutputs() const { return get_num_output(); }
  /*!\brief Return the number of input*/
  int NumInputs() const { return get_num_inputs(); }
  /*!\brief Use input index to set data to the runtime module.*/
  void SetInput(const int index, DLTensor* data_in) {
    NDArray input = get_input(index);
    DLTensor* dltensor_input = const_cast<DLTensor*>(input.operator->());
    CopyFromTo(data_in, dltensor_input);
  }
  /*!\brief Use the input name to set dat ato the runtime moduel. */
  void SetInput(const std::string name, DLTensor* data_in) {
    int index = this->GetInputIndex(name);
    SetInput(index, data_in);
  }
  /*!\brief Use output index to get a module output.*/
  NDArray GetOutput(int index) { return get_output(index); }
  /*!\brief Run the runtime.*/
  void Run() { run(); }
  /*!\brief Use the index to a module input.*/
  NDArray GetInput(int index) const { return get_input(index); }
  /*!\bief Use a input name to get the corresponding index of input.*/
  int GetInputIndex(const std::string& name) { return get_input_index(name); }
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
};      // namespace runtime
};      // namespace tvm
#endif  //  TVM_RUNTIME_PIPELINE_PIPELINE_STRUCT_H_
