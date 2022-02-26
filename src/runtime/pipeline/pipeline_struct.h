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

#include <atomic>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
namespace tvm {
namespace runtime {
#define GLOBAL_MODULE_INDEX -1
/*!
 *\brief The function is used to build the binding configuration for a runtime. The first
 * 'int' is the output index of the current runtime, the second 'int' is the index of child
 * runtime, and the 'string' is the input name of child runtime.
 */
using BindingConfigParseFunc = std::function<void(int, int, std::string)>;
/*!
 *\brief The 'pair' includes the module output index and the global output index.
 * The first 'int' is the module output index, and the second 'int' is the global output index.
 */
using GlobalOutputPair = std::pair<int, int>;
/*!
 *\brief The pair includes the module index and the module output index.
 * The first 'int' is the module index, and the second 'int' is the module output index.
 */
using ModuleOutputPair = std::pair<int, int>;
/*!
 *\brief The pair includes the runtime module index and the module input index.
 * The first 'int' is the module index, and the second 'int' is the module input index.
 */
using ModuleInputPair = std::pair<int, int>;
/*!\brief The runtime module interface type.*/
enum InterfaceType {
  INPUT = 0,
  OUTPUT,
};
/*!
 *\brief The structure includes the module index and the module output index.
 */
struct ModuleInterfaceID {
  ModuleInterfaceID() : runtime_idx(0), runtime_interface_idx(0), interface_type(OUTPUT) { ; }
  ModuleInterfaceID(int runtime_index, int runtime_interface_index, InterfaceType type = OUTPUT) {
    runtime_idx = runtime_index;
    runtime_interface_idx = runtime_interface_index;
    interface_type = type;
  }
  int runtime_idx;
  union {
    /*!\brief The output interface index.*/
    int runtime_output_idx;
    /*!\brief The input interface index.*/
    int runtime_input_idx;
    /*!\brief The interface index.*/
    int runtime_interface_idx;
  };
  /*!\brief The interface type*/
  InterfaceType interface_type;
};
/*!\brief The data notification structure.*/
class DataNotify {
 private:
  /*!\brief The 'contitional variable' is used to wait for notification.*/
  std::condition_variable notify_cv_;
  /*!\brief The mutex is used to protect the 'conditional variable'.*/
  std::mutex mutex_;
  /*!\brief Whether a data is ready or not.*/
  bool data_ready_ = false;
  /*!\brief Whether the thread should exit or not.*/
  std::atomic<bool> exit_state_{false};
  /*!
   * \brief The 'ModuleInterfaceID' in which the data was ready and triggered this
   *  notification.
   */
  ModuleInterfaceID notification_source_;

 public:
  /*!
   * \brief Constructing the DataNotify class.
   * \param parent_output_id The id of a runtime interface which is sending out the data
   *  notification.
   */
  explicit DataNotify(ModuleInterfaceID parent_output_id) {
    notification_source_ = parent_output_id;
  }
  /*!
   * \brief Getting the notification source.
   * \return The first 'int' is the runtime index, and the second 'int' is the output index.
   */
  ModuleInterfaceID GetNotifySource(void) { return notification_source_; }
  /*!
   *\brief Waiting for the notification.
   *\return Returning the value 'false' when the notification is in a 'exit' state, else
   * return true.
   */
  bool Wait(void) {
    std::unique_lock<std::mutex> lock(mutex_);
    notify_cv_.wait(lock, [&] { return this->data_ready_; });
    data_ready_ = false;
    return !GetExitState();
  }
  /*!brief Sending the notification in which the related data is ready.*/
  void Notify(void) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      data_ready_ = true;
    }
    notify_cv_.notify_one();
  }
  /*!brief Sending the notification when the notification state changes into 'exit'.*/
  void ExitNotify(void) {
    exit_state_.store(true, std::memory_order_release);
    Notify();
  }
  /*!
   *\brief Getting the 'exit state'.
   *\return Returning the value of 'exit_state_'
   */
  bool GetExitState(void) { return exit_state_.load(std::memory_order_acquire); }
};
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
   * \brief Enumerating the binding configuration.
   * \param parse_function The function is used to parse the binding configuration.
   * \param output_idx The index of output interface is used for parsing.
   */
  void VisitOutput(BindingConfigParseFunc parse_function, int output_idx) {
    for (auto output : bindings_) {
      parse_function(output_idx, output.first, output.second);
    }
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
  /*!
   * \brief Enumerating the output configuration.
   * \param parse_function The callback function is used to parse the binding configeration.
   */
  void VisitOutputConfig(BindingConfigParseFunc parse_function) {
    for (auto output : output_binding_map_) {
      output.second.VisitOutput(parse_function, output.first);
    }
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
  /*!
   * \brief Enumerating the binding configuration for a specified runtime.
   * \param parse_function The callback function is used to parse the binding configuration.
   * \param runtime_index The index of a runtime is used to parse the binding configuration.
   */
  void VisitRuntimeOutputConfig(BindingConfigParseFunc parse_function, int runtime_index) {
    auto config = config_.find(runtime_index);
    if (config == config_.end()) {
      LOG(FATAL) << "Do not finding the runtime " << runtime_index;
    }
    config->second.VisitOutputConfig(parse_function);
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
  using ModuleInputPairList = std::vector<std::pair<std::shared_ptr<BackendRuntime>, int>>;

 private:
  /*\brief The index of runtime indicates the runtime position in the pipeline.*/
  int runtime_idx_;
  /*\brief The Runtime module of a backend graph executor.*/
  Module module_;
  /*\brief The thread is associated with the current runtime*/
  std::thread thread_;
  /*\brief A list of runtime which depends on the current runtime.*/
  std::unordered_map<int, ModuleInputPairList> children_;
  /*\brief A map including the runtime input index and the notification data structure.*/
  std::unordered_map<int, std::shared_ptr<DataNotify>> parents_notify_;
  /*\brief The execution count of the 'RunPipeline' function. */
  uint32_t pipeline_execution_count_ = 0;
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
  /*!\brief The worker thread is used to execute the runtimes in pipeline.*/
  void StartWorkThread() {
    if (runtime_idx_ == 0) {
      this->CreateParentsNotify(0, GLOBAL_MODULE_INDEX, 0);
    } else {
      // Only launching the worker thread for the runtimes after the first runtime.
      thread_ = std::thread([&]() {
        while (!this->WaitAndLoadPipelineData()) {
          this->RunPipeline();
        }
        VLOG(1) << "Runtime " << this->runtime_idx_ << " exit.";
      });
    }
    return;
  }
  /*!\brief Stopping the threads in pipeline.*/
  void StopPipeline() {
    for (auto notify : parents_notify_) {
      notify.second->ExitNotify();
    }
    if (thread_.joinable()) {
      thread_.join();
    }
  }
  /*!
   * \brief Waiting for the internal forwarding data.
   * \return Returning 'true' when getting a 'exit' notification otherwise returning 'false'.
   */
  bool WaitAndLoadPipelineData() {
    std::unordered_map<int, std::shared_ptr<DataNotify>> notifys = parents_notify_;
    bool exit_notify = false;
    while (!notifys.empty() && !exit_notify) {
      auto notify = notifys.begin();
      // Breaking the loop when the notification is in the exit state.
      if ((exit_notify = notify->second->GetExitState())) break;
      // Getting the source which sends this notification.
      auto notify_source = notify->second->GetNotifySource();
      // Loading the binding data.
      while (!this->LoadBindingData(notify->first, notify_source.runtime_idx,
                                    notify_source.runtime_output_idx)) {
        // Waiting for the notification.
        if (!notify->second->Wait()) {
          VLOG(1) << "runtime index:" << runtime_idx_ << " receive exit notify.";
          exit_notify = true;
          break;
        }
        // TODO(huajsj): removing this 'break' after finishing the 'LoadBindingData'.
        break;
      }
      VLOG(1) << "runtime_index.input_index:" << runtime_idx_ << "." << notify->first
              << "from runtime_index.output_index:" << notify_source.runtime_idx << "."
              << notify_source.runtime_output_idx;
      notifys.erase(notify);
    }
    return exit_notify;
  }
  /*!
   * \brief Loading the binding data.
   * \param parent_idx The index of runtime which forwards data to current runtime.
   * \param parent_output_idx The index of output where the forwarding data is coming from.
   * \param input_idx The index of input where the data will be forwarding to.
   * \return Returning 'true' when data is loaded successfully, otherwise returning 'false'.
   */
  bool LoadBindingData(int parent_idx, int parent_output_idx, int input_idx) {
    // TODO(huajsj): Loading data.
    return false;
  }
  /*!
   * \brief Forwarding the output data into the child runtimes.
   */
  void ForwardingOutputDataToChildren(void) {
    for (auto child : children_) {
      // TODO(huajsj): Getting the output data from the current runtime in order to forward
      // data to the child.

      // Notifying the 'children runtime' that the forwarding data are ready.
      for (auto module_pair : child.second) {
        module_pair.first->ParentNotify(module_pair.second);
      }
    }
  }
  /*!
   *\brief Creating a parent notification.
   *\param input_index The input index of the 'current runtime'.
   *\param parent_idx The index of 'parent runtime' which will send the notification.
   *\param parent_output_idx The output index of the 'parent runtime' which will send
   * the nofication.
   */
  void CreateParentsNotify(int input_index, int parent_idx, int parent_output_idx) {
    if (parents_notify_.find(input_index) != parents_notify_.end()) {
      LOG(FATAL) << "Not finding the input index " << input_index << " in runtime " << runtime_idx_;
    }
    parents_notify_[input_index] =
        std::make_shared<DataNotify>(ModuleInterfaceID(parent_idx, parent_output_idx));
  }
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
  ~BackendRuntime() {
    for (auto data : input_tensor_local_copy_) {
      TVMArrayFree(data.second);
    }
    StopPipeline();
  }
  /*!brief Getting the runtime index*/
  int GetIndex() const { return runtime_idx_; }
  /*!
   * \brief Getting the times of using pipeline function.
   * \return The times of using pipeline function.
   */
  int GetExecutionCount() const { return pipeline_execution_count_; }
  /*!
   * \brief Initializing data structures for the pipeline execution.
   * \param config The pipeline configueration.
   * \param runtimes A list of BackendRuntime.
   */
  void InitializePipeline(ConfigPipelineExecution config,
                          std::vector<std::shared_ptr<BackendRuntime>>* runtimes) {
    // Getting the 'binding configuration' for each runtime.
    config.VisitRuntimeOutputConfig(
        [&](int output_idx, int child_idx, std::string child_input_name) {
          int runtime_idx_max = runtimes->size();
          if (child_idx < 0 || child_idx >= runtime_idx_max) {
            LOG(FATAL) << "The runtime index " << child_idx << " is out of the range.";
          }
          auto child_runtime = runtimes->at(child_idx);
          int input_index = child_runtime->GetInputIndex(child_input_name);
          if (input_index < 0) {
            LOG(FATAL) << "Can not find the input " << input_index << "in runtime " << child_idx;
          }
          children_[output_idx].push_back(std::make_pair(child_runtime, input_index));
          child_runtime->CreateParentsNotify(input_index, runtime_idx_, output_idx);
          VLOG(1) << " parent_idx.output:" << runtime_idx_ << "." << output_idx << " child.input"
                  << child_idx << "." << input_index;
        },
        runtime_idx_);

    StartWorkThread();
  }
  /*!
   * \brief Notifying a input is ready.
   * \param input_index The index of 'input interface' which is ready for data.
   */
  void ParentNotify(int input_index) {
    auto notify = parents_notify_.find(input_index);
    if (notify == parents_notify_.end()) {
      LOG(FATAL) << "Can not find the input for index " << input_index << " in runtime"
                 << runtime_idx_;
      return;
    }
    notify->second->Notify();
    VLOG(1) << "Notification at runtime_index.input_index:" << runtime_idx_ << "." << input_index;
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
  /*!\brief Running the runtime in the pipeline mode.*/
  void RunPipeline() {
    Run();
    ForwardingOutputDataToChildren();
    pipeline_execution_count_++;
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
};      // namespace runtime
};      // namespace tvm
#endif  //  TVM_RUNTIME_PIPELINE_PIPELINE_STRUCT_H_
