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
 *\brief The pair includes the module output index and the global output index.
 * The first 'int' is the module output index, and the second 'int' is the global output index.
 */
using GlobalOutputPair = std::pair<int, int>;
/*!
 *\brief The pair includes the module index and the module output index.
 * The first 'int' is the module index, and the second 'int' is the module output index.
 */
using ModuleOutputPair = std::pair<int, int>;
/*!
 * \brief All binding information of a output interface.
 */
class ConfigBindings {
 public:
  /*!\brief Whether this binding is bound to the PipelineExecutor output interface.*/
  bool IsGlobalOutput() const { return global_output_index_ > -1; }
  /*!\brief Getting the global output index in the current binding.*/
  int GetGlobalOutputIndex() const { return global_output_index_; }
  /*!\brief Returning the binding configuration.*/
  std::unordered_map<int, std::string>& Get() { return bindings_; }
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
   *\brief Getting the map which includes the global outputs and the current module outputs.
   *\return A list of "GlobalOutputPair".
   */
  std::vector<GlobalOutputPair> GetGlobalConfigOutputBindings(void) const {
    std::vector<GlobalOutputPair> ret;
    for (auto bindings : output_binding_map_) {
      if (bindings.second.IsGlobalOutput()) {
        ret.push_back(GlobalOutputPair(bindings.first, bindings.second.GetGlobalOutputIndex()));
      }
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
  ConfigOutputBindings& operator[](int key) {
    ICHECK(config_.find(key) != config_.end());
    return config_[key];
  }
  /*
   *!\brief This function is used to verify whether config is loaded successfully.
   * \return Return "true" to indicate that this class has not been successfully loaded.
   */
  bool Empty() { return config_.empty(); }
  /*!
   *\brief Check if the module index existing in the "config".
   */
  bool FindModuleInConfig(int mod_idx) { return config_.find(mod_idx) != config_.end(); }
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
  /*
   *!\brief Get the map of global outputs and module outputs.
   */
  std::unordered_map<int, ModuleOutputPair> GetGlobalConfigOutputBindings(void) const {
    return global_output_map_;
  }
  /*
   *!\brief Parsing the configuration.
   */
  void ParseConfiguration(const std::unordered_map<int, ConfigOutputBindings>& config) {
    if (config.empty()) {
      LOG(FATAL) << "The Configuration loading not finish yet.";
    }
    for (auto mod_output : config) {
      // Using the global output index as the key to create a map including global index and
      // module output index.
      const std::vector<GlobalOutputPair>& global_output =
          mod_output.second.GetGlobalConfigOutputBindings();

      for (auto output : global_output) {
        global_output_map_[output.second] = ModuleOutputPair(mod_output.first, output.first);
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
      // Build the mapping of mod_idx and "ConfigOutputBindings".
      config_[mod_idx] = output;
    }
    // Doing the configuration parsing after the loading finished.
    ParseConfiguration(config_);
  }

 private:
  /*
   *!\brief The key is the module index, this variable records all module pipeline configuration
   * information.
   */
  std::unordered_map<int, ConfigOutputBindings> config_;
  /*
   *\brief The key is the global output index, and the map is including global outputs index and
   * the module outputs pair.
   */
  std::unordered_map<int, ModuleOutputPair> global_output_map_;
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
 * \brief A map includes global module parameters groups and graph modudles.
 */
struct ParamConnectionConfig {
  /*!\brief Mapping from the name of a global module parameters group to the index of a runtime
   *  module.
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
   * \brief Load from JSONReader.
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
      ICHECK(mod_idx >= 0) << "Invalid module index value " << mod_idx;
      ICHECK(!global_param_name.empty()) << "Invalid global parameter group name value";
      param_connection[global_param_name] = mod_idx;
    }
  }
};
/*
 *\brief Backend Runtime.
 */
class BackendRuntime {
 private:
  /*\brief The index of runtime indicates the runtime position in the pipeline.*/
  int runtime_idx_;
  /*\brief The Runtime module of a backend graph executor.*/
  Module module_;
  /*!
   *\brief In order to transfer data from one backend runtime to another, we need a local
   * tensor variable as a medium. "input_tensor_local_copy_" is a map including
   * input data and local tensor vairable.
   */
  std::unordered_map<DLTensor*, DLTensor*> input_tensor_local_copy_;
  /*!\brief The packed functions.*/
  tvm::runtime::PackedFunc set_input_;
  tvm::runtime::PackedFunc get_input_;
  tvm::runtime::PackedFunc get_output_;
  tvm::runtime::PackedFunc get_num_output_;
  tvm::runtime::PackedFunc get_num_inputs_;
  tvm::runtime::PackedFunc get_input_index_;
  tvm::runtime::PackedFunc run_;
  /*!
   * \brief Copying from a given tensor and using 'CPU' as the device.
   */
  inline DLTensor* CopyDLTensorToCPU(const DLTensor* from) {
    DLTensor* ret = NULL;
    TVMArrayAlloc(from->shape, from->ndim, from->dtype.code, from->dtype.bits, from->dtype.lanes,
                  kDLCPU, 0, &ret);
    return ret;
  }
  /*!\brief Creating a new NDArray with same shape and data type as the given DLTensor.*/
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
   *\brief Copying data from one DLTensor to another.
   */
  void CopyFromTo(DLTensor* from, DLTensor* to) {
    // When the 'from' device and the 'to' device are not the same, we use a temporary CPU
    // DLTensor as the bridge.
    if (from->device.device_type != to->device.device_type && from->device.device_type != kDLCPU &&
        to->device.device_type != kDLCPU) {
      DLTensor* dltensor_local = nullptr;
      if (input_tensor_local_copy_.find(to) == input_tensor_local_copy_.end()) {
        dltensor_local = CopyDLTensorToCPU(from);
        input_tensor_local_copy_[to] = dltensor_local;
      } else {
        dltensor_local = input_tensor_local_copy_[to];
      }
      TVMArrayCopyFromTo(from, dltensor_local, nullptr);
      from = dltensor_local;
    }

    TVMArrayCopyFromTo(from, to, nullptr);
  }

 public:
  BackendRuntime(Module mod, int mod_idx) {
    module_ = mod;
    runtime_idx_ = mod_idx;
    get_input_index_ = module_.GetFunction("get_input_index");
    get_num_output_ = module_.GetFunction("get_num_outputs");
    get_num_inputs_ = module_.GetFunction("get_num_inputs");
    set_input_ = module_.GetFunction("set_input");
    get_input_ = module_.GetFunction("get_input");
    get_output_ = module_.GetFunction("get_output");
    run_ = module_.GetFunction("run");
  }
  BackendRuntime(void) {}
  ~BackendRuntime() {
    for (auto data : input_tensor_local_copy_) {
      TVMArrayFree(data.second);
    }
  }
  /*!\brief Creating a NDArray containing same shape and data type with a module output. */
  NDArray CreateFromOutput(int idx) {
    NDArray data = get_output_(idx);
    return CreateNDArrayFromDLTensor(const_cast<DLTensor*>(data.operator->()));
  }
  /*!\brief Return the index of the current module.*/
  int GetModuleIndex() { return runtime_idx_; }
  /*!\brief Return the number of output*/
  int NumOutputs() const { return get_num_output_(); }
  /*!\brief Return the number of input*/
  int NumInputs() const { return get_num_inputs_(); }
  /*!\brief Setting the data to this module via input index.*/
  void SetInput(const int index, DLTensor* data_in) {
    NDArray input = get_input_(index);
    DLTensor* dltensor_input = const_cast<DLTensor*>(input.operator->());
    CopyFromTo(data_in, dltensor_input);
  }
  /*!\brief Setting the data to the current runtime moduel via the input name. */
  void SetInput(const std::string name, DLTensor* data_in) {
    int index = this->GetInputIndex(name);
    SetInput(index, data_in);
  }
  /*!\brief Getting the input data via the input index.*/
  NDArray GetInput(int index) const { return get_input_(index); }
  /*!\bief Getting the input data via the input name.*/
  int GetInputIndex(const std::string& name) { return get_input_index_(name); }
  /*!\brief Using the output index to get the module output.*/
  NDArray GetOutput(int index) { return get_output_(index); }
  /*!\brief Running the runtime.*/
  void Run() { run_(); }
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
