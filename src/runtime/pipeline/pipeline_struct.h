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
#include <tvm/runtime/threading_backend.h>

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

#include "spsc_queue.h"
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
/*!\brief The state of the pipeline.*/
enum PipelineState {
  STOPPED = 0,
  RUNNING,
  STOPPING,
};
/*!
 *\brief The structure includes the module index and the module output index.
 */
struct ModuleInterfaceID {
  ModuleInterfaceID() { SetID(0, 0, INPUT); }
  ModuleInterfaceID(int runtime_index, int runtime_interface_index, InterfaceType type = INPUT) {
    SetID(runtime_index, runtime_interface_index, type);
  }
  /*!
   * \brief Set the value of ID.
   * \param runtime_index The index of runtime.
   * \param runtime_interface_index The index of interface.
   * \param type The type of the interface.
   */
  void SetID(int runtime_index, int runtime_interface_index, InterfaceType type) {
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
  ModuleInterfaceID& operator=(const struct ModuleInterfaceID& id) {
    SetID(id.runtime_idx, id.runtime_interface_idx, id.interface_type);
    return *this;
  }
  bool operator==(const struct ModuleInterfaceID& id) const {
    return id.interface_type == interface_type &&
           id.runtime_interface_idx == runtime_interface_idx && id.runtime_idx == runtime_idx;
  }
};
/*!brief The hash function used to generate the hash value for the "ModuleInterfaceID" variable.*/
struct ModuleIDHash {
  bool operator()(const ModuleInterfaceID& id) const {
    int offset = sizeof(std::size_t) / 3;
    return id.interface_type | id.runtime_interface_idx << offset | id.runtime_idx << offset * 2;
  }
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
  /*!\brief The 'ModuleInterfaceID' of an interface which sent this notification.*/
  ModuleInterfaceID notification_source_;

 public:
  /*!
   * \brief Constructing the DataNotify class.
   * \param source_interface_id The id of a runtime interface which is sending out the data
   *  notification.
   */
  explicit DataNotify(ModuleInterfaceID source_interface_id) {
    notification_source_ = source_interface_id;
  }
  /*!
   * \brief Getting the notification target.
   * \return The ID of the interface which is sending out the notification.
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
/*!\brief The container used to store the forwarding data of the pipeline.*/
class QueueData {
 public:
  explicit QueueData(DLTensor* data) {
    if (data_ == data) {
      LOG(FATAL) << "The value of 'data'(" << data << ") is the same as 'data_'(" << data_ << ")";
    }
    data_ = data;
    SetAsDataOwner(false);
  }
  QueueData() { SetAsDataOwner(true); }
  /*!\brief Doing a deep copy for the 'QueueData' structure.*/
  QueueData& operator=(const QueueData& data) {
    CreateCopyFrom(data.GetDLData());
    return *this;
  }
  QueueData& operator=(const NDArray& from) {
    CreateCopyFrom(const_cast<DLTensor*>(from.operator->()));
    return *this;
  }
  QueueData& operator=(const DLTensor* from) {
    CreateCopyFrom(from);
    return *this;
  }
  /*!\brief Create a deep copy of the 'DLTensor' data.*/
  DLTensor* CreateCopyFrom(const DLTensor* from) {
    if (!from) {
      LOG(FATAL) << "the 'from' pointer is a null pointer!";
    }
    size_t fromLen = tvm::runtime::GetDataSize(*from);
    size_t toLen = data_ ? tvm::runtime::GetDataSize(*data_) : 0;
    if (fromLen != toLen) {
      // If this container ownes the variable 'data_', then recreating the 'data_' variable.
      if (IsDataOwner()) {
        if (data_) {
          TVMArrayFree(data_);
          data_ = nullptr;
        }
        TVMArrayAlloc(from->shape, from->ndim, from->dtype.code, from->dtype.bits,
                      from->dtype.lanes, from->device.device_type, from->device.device_id, &data_);
      } else {
        LOG(FATAL) << "The 'from' data is not matched with the  'data_'.";
      }
    }
    TVMArrayCopyFromTo(const_cast<DLTensor*>(from), data_, nullptr);
    return data_;
  }
  /*!\brief Return a pointer to the 'DLTensor' data.*/
  DLTensor* GetDLData() const { return data_; }
  ~QueueData() {
    if (IsDataOwner() && data_) {
      TVMArrayFree(data_);
      data_ = nullptr;
    }
  }

 private:
  /*!\brief Pointer to the forwarding data.*/
  DLTensor* data_ = nullptr;
  /*!\brief Whether this container is the owner of the 'data_'.*/
  bool is_data_owner_ = false;
  /*!\brief Set the current container as the owner of the 'data_'.*/
  void SetAsDataOwner(bool is_owner) { is_data_owner_ = is_owner; }
  /*!Check whether the current container is the owner of the 'data_'.*/
  bool IsDataOwner() const { return is_data_owner_; }
};
/*!
 * \brief All binding information of an output interface.
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
    if (IsGlobalOutput()) {
      parse_function(output_idx, GLOBAL_MODULE_INDEX, std::to_string(global_output_index_));
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
class ConfigRuntime {
 public:
  ConfigRuntime& operator=(const ConfigRuntime& output) {
    output_binding_map_ = output.GetOutBindings();
    cpu_affinity_ = output.GetCPUAffinity();
    return *this;
  }

  ConfigBindings& operator[](const int key) {
    ICHECK(output_binding_map_.find(key) != output_binding_map_.end());
    return output_binding_map_[key];
  }
  /*!
   * \brief Store the CPU affinity settings.
   * \param cpu_affinity The CPU affinity settings in the text form.
   */
  void StoreCPUAffinity(std::string cpu_affinity) { cpu_affinity_ = cpu_affinity; }
  /*!
   * \brief Getting the setting of the cpu affinity.
   * \param Returning the cpu affinity in text form.
   */
  std::string GetCPUAffinity() const { return cpu_affinity_; }
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
   *\brief This function is used to verify whether ConfigRuntime is successfully loaded.
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
   * \brief Create an output binding map from JSONReader.
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
  /*!\brief The cpu affinity setting for the tvm thread pool.*/
  std::string cpu_affinity_;
};

/*!
 * \brief The binding or dependency information of each module output interface.
 */
class ConfigPipelineExecution {
 public:
  ConfigRuntime& operator[](int key) {
    ICHECK(config_.find(key) != config_.end());
    return config_[key];
  }
  /*Get the cpu affinity settings.*/
  std::string GetCPUAffinity(int runtime_idx) {
    auto config = config_.find(runtime_idx);
    if (config == config_.end()) {
      LOG(FATAL) << "Do not finding the runtime " << runtime_idx;
    }
    auto config_runtime = config->second;
    return config_runtime.GetCPUAffinity();
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
  void ParseConfiguration(const std::unordered_map<int, ConfigRuntime>& config) {
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
      ConfigRuntime output;
      std::string dev;
      std::string cpu_affinity;
      while (reader->NextObjectItem(&key)) {
        if (key == "mod_idx") {
          reader->Read(&mod_idx);
        } else if (key == "dev") {
          reader->Read(&dev);
        } else if (key == "output") {
          reader->Read(&output);
        } else if (key == "cpu_affinity") {
          reader->Read(&cpu_affinity);
        } else {
          LOG(FATAL) << "do not support key " << key;
        }
      }
      ICHECK(mod_idx >= 0) << "Invalid mod_idx value " << mod_idx;
      // Check if the output is successfully read.
      ICHECK(!output.Empty()) << "Invalid output binding result.";
      // Store the cpu affinity into the 'ConfigRuntime' structure.
      output.StoreCPUAffinity(cpu_affinity);
      // Build the mapping of mod_idx and "ConfigRuntime".
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
  std::unordered_map<int, ConfigRuntime> config_;
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
  /*!\brief The map includes the global input name and global input index.*/
  std::unordered_map<std::string, int> input_name_index_map;
  /*!
   * \brief The map not only includes the runtime index ,but also the pair of global interface
   *  and runtime interface.
   */
  std::unordered_map<int, std::vector<std::pair<std::string, std::string>>> input_runtime_map;
  std::pair<int, std::string> operator[](const std::string key) {
    if (input_connection.find(key) == input_connection.end()) {
      LOG(FATAL) << "Not find the key " << key;
    }
    return input_connection[key];
  }
  /*!\brief Returns the number of global inputs through the input_runtime_map list size.*/
  int GetInputNum() { return input_runtime_map.size(); }

  /*!
   * \brief Getting the global input index through the input name.
   * \param input_name The global input name.
   */
  int GetInputIndex(std::string input_name) {
    auto input_index_iter = input_name_index_map.find(input_name);
    if (input_index_iter == input_name_index_map.end()) {
      LOG(FATAL) << "Do not finding the input name! " << input_name;
    }
    return input_index_iter->second;
  }
  /*!\brief Enumerating the input binding configuration for a specified runtime.*/
  void VisitConfig(BindingConfigParseFunc parse_function, int runtime_index) {
    auto config = input_runtime_map.find(runtime_index);
    // Only do the processing when there are input configuration in the runtime.
    if (config != input_runtime_map.end()) {
      for (auto x : config->second) {
        int input_index = GetInputIndex(x.first);
        parse_function(input_index, runtime_index, x.second);
      }
    }
  }
  /*!
   * \brief Create an input connection config from JSONReader.
   * \param reader Json reader.
   */
  void Load(dmlc::JSONReader* reader) {
    int global_interface_index = 0;
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
          input_name_index_map[global_interface_name] = global_interface_index++;
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
      // Creating a map which not only includes the runtime index, but also the pair of gloal
      // interface, and runtime interface.
      input_runtime_map[mod_idx].push_back(
          std::make_pair(global_interface_name, module_interface_name));
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
/*!
 * \brief The single consumer single producer queue which is used to forward data between two
 * interfaces of backend cores.
 */
using ForwardQueue = SPSCLockFreeQueue<QueueData, ModuleInterfaceID>;
using ForwardQueueMap =
    std::unordered_map<ModuleInterfaceID, std::shared_ptr<ForwardQueue>, ModuleIDHash>;
/*!\brief The basic class for runtime.*/
class BasicRuntime {
  using ModuleInputPairList = std::vector<std::pair<std::shared_ptr<BasicRuntime>, int>>;

 public:
  explicit BasicRuntime(int runtime_idx) : runtime_idx_(runtime_idx) {}
  /*!\brief Return the index of the current module.*/
  int GetModuleIndex() { return runtime_idx_; }
  /*!\brief Setting the data into this runtime via the input index.*/
  virtual void SetInput(const int index, DLTensor* data_in) {}
  /*!
   * \brief Sending a notification when data is ready.
   * \param input_index The index of an input interface which have data ready.
   */
  virtual void ParentNotify(int input_index) {}
  /*!
   *\brief Creating a parent notification.
   *\param input_index The input index of the 'current runtime'.
   *\param parent_idx The index of 'parent runtime' which will send the notification.
   *\param parent_output_idx The output index of the 'parent runtime' which will send
   * the notification.
   */
  void CreateParentsNotify(int input_index, int parent_idx, int parent_output_idx) {
    if (parents_notify_.find(input_index) != parents_notify_.end()) {
      LOG(FATAL) << "The notification associated with the input interface " << input_index
                 << " in runtime " << runtime_idx_ << " already been created!";
      return;
    }
    parents_notify_[input_index] =
        std::make_shared<DataNotify>(ModuleInterfaceID(parent_idx, parent_output_idx, OUTPUT));
  }

 protected:
  /*!\brief The index of runtime indicates the runtime position in the pipeline.*/
  int runtime_idx_;
  /*!\brief A list of runtime which depends on the current runtime.*/
  std::unordered_map<int, ModuleInputPairList> children_;
  /*!\brief The map includes the runtime input index and the notification data structure.*/
  std::unordered_map<int, std::shared_ptr<DataNotify>> parents_notify_;
  /*!
   * \brief There is a list of SPSC input queues in which the input interface would poll the
   *  data comed from other backend cores.
   */
  std::unordered_map<int, std::shared_ptr<ForwardQueue>> input_queue_;

  /*!
   * \brief A list of SPSC forward queues in which the parent interface will push the data to
   *  other backend cores.
   */
  std::unordered_map<int, ForwardQueueMap> forward_queue_;
  /*!\brief The state of the pipeline.*/
  std::atomic<PipelineState> pipeline_state_{STOPPED};
  /*!
   * \brief Generate the ID of an input queue.
   * \param runtime_index The index of backend runtime.
   * \param interface_index The index of the interface.
   * \param type The type of the interface.
   */
  ModuleInterfaceID GenerateQueueID(int runtime_index, int interface_index, InterfaceType type) {
    return ModuleInterfaceID(runtime_index, interface_index, type);
  }
  /*!
   * \brief Forwarding the data into the child runtimes.
   * \param forward_queue_map The map includes the id and the queue.
   * \param child_runtime The child runtime.
   * \param child_input_index The child runtime index.
   * \param data The data is used for forwarding.
   */
  bool ForwardData(const ForwardQueueMap* forward_queue_map,
                   std::shared_ptr<BasicRuntime> child_runtime, int child_input_index,
                   const DLTensor* data) {
    auto child_runtime_index = child_runtime->GetModuleIndex();
    auto queue_id = GenerateQueueID(child_runtime_index, child_input_index, INPUT);
    if (forward_queue_map->find(queue_id) == forward_queue_map->end()) {
      LOG(FATAL) << "Not find the associated queue of the runtime(" << child_runtime_index
                 << ").input(" << child_input_index << ") which is connected with runtime("
                 << runtime_idx_;
    }
    auto forward_queue = forward_queue_map->at(queue_id);
    // If the queue is full, keep try until the push get success or the pipeline run into
    // a STOP state.
    while (!forward_queue->Push<const DLTensor*>(data)) {
      if (PipelineIsStop()) {
        LOG(INFO) << "The forwarding process is stopped after the pipeline status is changed"
                  << " into stop.";
        return false;
      }
    }
    child_runtime->ParentNotify(child_input_index);
    return true;
  }
  /*!
   * \brief Creating a forwarding queue for the pair of an output interface and an input interface.
   * \param forward_inf_idx The index of an interface which will send the forwarding data.
   * \param child_runtime The backend runtime which owns the input interface.
   * \param input_index The index of an input interface. This interface will receive the
   * forwarding data.
   */
  void CreateForwardingQueue(int forward_inf_idx, std::shared_ptr<BasicRuntime> child_runtime,
                             int input_index) {
    auto queue_id = GenerateQueueID(child_runtime->GetModuleIndex(), input_index, INPUT);
    // The forwarding queue map of a specified output interface.
    auto& queue_map = forward_queue_[forward_inf_idx];
    if (queue_map.find(queue_id) != queue_map.end()) {
      LOG(FATAL) << "The queue " << queue_id.runtime_idx << "." << queue_id.runtime_interface_idx
                 << " is already created!";
      return;
    }
    auto queue = std::make_shared<ForwardQueue>(queue_id);
    queue_map[queue_id] = queue;
    // Use the created queue as the consumer queue for the input interface of this forwarding
    // pair.
    child_runtime->AppendInputQueue(input_index, queue);
  }
  /*!
   * \brief Setting  the consumer queue for the input interface.
   * \param input_index The index of the input interface.
   * \param queue The consumer queue.
   */
  void AppendInputQueue(int input_index, std::shared_ptr<ForwardQueue> queue) {
    input_queue_[input_index] = queue;
  }
  /*!\brief Checking if the pipeline is stopped or stopping.*/
  const bool PipelineIsStop() const {
    auto state = pipeline_state_.load(std::memory_order_acquire);
    return state == STOPPING || state == STOPPED;
  }
};
/*
 *!\brief Backend Runtime.
 */
class BackendRuntime : public BasicRuntime {
 private:
  /*!The cpu affinity settings for this runtime.*/
  std::string cpu_affinity_ = "";
  /*!\brief The Runtime module of a backend graph executor.*/
  Module module_;
  /*\brief The thread is associated with the current runtime*/
  std::thread thread_;
  /*!\brief The execution count of the 'RunPipeline' function. */
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
    SetPipelineState(RUNNING);
    if (runtime_idx_ == 0) {
      this->SetCPUAffinity();
    } else {
      // Only launching the worker thread for the runtimes after the first runtime.
      thread_ = std::thread([&]() {
        this->SetCPUAffinity();
        while (!this->WaitAndLoadPipelineData()) {
          if (!this->RunPipeline()) {
            break;
          }
        }
        VLOG(1) << "Runtime " << this->runtime_idx_ << " exit.";
      });
    }
    return;
  }
  /*!\brief Setting the state of the pipeline.*/
  void SetPipelineState(PipelineState state) {
    pipeline_state_.store(state, std::memory_order_release);
  }
  /*!\brief Stopping the threads in pipeline.*/
  void StopPipeline() {
    SetPipelineState(STOPPING);
    for (auto notify : parents_notify_) {
      notify.second->ExitNotify();
    }
    if (thread_.joinable()) {
      thread_.join();
    }
    SetPipelineState(STOPPED);
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
      auto target_input_interface_index = notify->first;
      // Loading the binding data.
      while (!this->LoadBindingData(target_input_interface_index)) {
        // Waiting for the notification.
        if (!notify->second->Wait()) {
          exit_notify = true;
          break;
        }
      }
      notifys.erase(notify);
    }
    return exit_notify;
  }
  /*!
   * \brief Loading the binding data.
   * \param input_index The index of the interface which will receive the forwarding data.
   * \return Returning 'true' when data is loaded successfully, otherwise returning 'false'.
   */
  bool LoadBindingData(int input_index) {
    if (input_queue_.find(input_index) == input_queue_.end()) {
      LOG(FATAL) << "Not finding the associated input queue of the input " << input_index << " !";
    }
    auto queue = input_queue_[input_index];
    QueueData data;
    // TODO(huajsj): Doing the 'SetInput' inside the poll function to avoid one time data copy.
    if (!queue->Poll<QueueData>(&data)) {
      return false;
    }
    SetInput(input_index, data.GetDLData());
    return true;
  }
  /*!
   * \brief Forwarding the output data into the child runtimes.
   * \return bool Return false when the "PipelineIsStop" function returns true or this function
   *  reaches some errors. Otherwise, return true.
   */
  bool ForwardingOutputDataToChildren(void) {
    for (auto child : children_) {
      auto output_idx = child.first;
      if (forward_queue_.find(output_idx) == forward_queue_.end()) {
        LOG(FATAL) << "Not find the forwarding queue map for output(" << output_idx << ")!";
      }
      NDArray output = GetOutput(output_idx);
      auto forward_queue_map = forward_queue_[output_idx];
      // Notifying the 'children runtime' that the forwarding data are ready.
      for (auto module_pair : child.second) {
        auto child_runtime = module_pair.first;
        auto child_input_index = module_pair.second;
        auto output_data = const_cast<DLTensor*>(output.operator->());
        if (!ForwardData(&forward_queue_map, child_runtime, child_input_index, output_data)) {
          return false;
        }
      }
    }
    return true;
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
  /*!\brief Setting the cpu affinity for the tvm threads pool in the current BackendRuntime.*/
  void SetCPUAffinity(void) {
    if (cpu_affinity_.empty()) {
      return;
    }
    auto affinity_mode = tvm::runtime::threading::ThreadGroup::kSpecifyThreadShareAllCore;
    std::istringstream istr(cpu_affinity_);
    std::string affinity;
    std::vector<unsigned int> cpus;
    while (getline(istr, affinity, ',')) {
      cpus.push_back(std::stoi(affinity));
    }
    tvm::runtime::threading::Configure(affinity_mode, 0, cpus);
  }

 public:
  BackendRuntime(Module mod, int mod_idx) : BasicRuntime(mod_idx), module_(mod) {
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
                          std::vector<std::shared_ptr<BackendRuntime>>* runtimes,
                          std::shared_ptr<BasicRuntime> global_runtime) {
    // Getting the current BackendRuntime's cpu affinity setting.
    cpu_affinity_ = config.GetCPUAffinity(runtime_idx_);
    // Getting the 'binding configuration' for each child runtime.
    config.VisitRuntimeOutputConfig(
        [&](int output_idx, int child_idx, std::string child_input_name) {
          std::shared_ptr<BasicRuntime> child_runtime = nullptr;
          int input_index;
          if (GLOBAL_MODULE_INDEX == child_idx) {
            int global_output_index = std::stoi(child_input_name);
            input_index = global_output_index;
            child_runtime = global_runtime;
          } else {
            int runtime_idx_max = runtimes->size();
            if (child_idx < 0 || child_idx >= runtime_idx_max) {
              LOG(FATAL) << "The runtime index " << child_idx << " is out of the range.";
            }
            auto runtime = runtimes->at(child_idx);
            ICHECK(runtime->GetModuleIndex() == child_idx);
            input_index = runtime->GetInputIndex(child_input_name);
            if (input_index < 0) {
              LOG(FATAL) << "Can not find the input " << input_index << "in runtime " << child_idx;
            }
            child_runtime = runtime;
          }
          ICHECK(child_runtime != nullptr);
          children_[output_idx].push_back(std::make_pair(child_runtime, input_index));
          child_runtime->CreateParentsNotify(input_index, runtime_idx_, output_idx);
          VLOG(1) << " parent_idx.output:" << runtime_idx_ << "." << output_idx
                  << " child.input:" << child_idx << "." << input_index;
          // Creating the pipeline forwarding queue.
          this->CreateForwardingQueue(output_idx, child_runtime, input_index);
        },
        runtime_idx_);

    StartWorkThread();
  }
  /*!
   * \brief Notifying an input is ready.
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
  }
  /*!\brief Creating a NDArray containing same shape and data type with a module output. */
  NDArray CreateFromOutput(int idx) {
    NDArray data = get_output_(idx);
    return CreateNDArrayFromDLTensor(const_cast<DLTensor*>(data.operator->()));
  }
  /*!\brief Return the number of output*/
  int NumOutputs() const { return get_num_output_(); }
  /*!\brief Return the number of input*/
  int NumInputs() const { return get_num_inputs_(); }
  /*!\brief Setting the data to this runtime via input index.*/
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
  /*!
   * \brief Running the runtime in the pipeline mode.
   * \return Returning false if the forwarding function failed. Otherwise, returning true.;
   */
  bool RunPipeline() {
    Run();
    bool ret = ForwardingOutputDataToChildren();
    pipeline_execution_count_++;
    return ret;
  }
};
/*!
 * \brief This global runtime represents the pipeline executor and exposes the input and output
 *  interface.
 */
class GlobalRuntime : public BasicRuntime {
 public:
  explicit GlobalRuntime(int runtime_idx) : BasicRuntime(runtime_idx) {}
  /**/
  std::vector<std::shared_ptr<BackendRuntime>> GetRuntimeList() { return runtimes_; }
  /*!\brief Push the data into the queue for the current runtime.*/
  void SetPipelineInput(const std::string input_name, DLTensor* data_in) {
    auto input_index = input_config_.GetInputIndex(input_name);
    auto child_iter = children_.find(input_index);
    if (child_iter == children_.end()) {
      return;
    }
    auto forward_queue_map = forward_queue_[input_index];
    // Notifying the 'children runtime' that the forwarding data are ready.
    for (auto module_pair : child_iter->second) {
      auto child_runtime = module_pair.first;
      auto child_input_index = module_pair.second;
      // No need to go through the forward queue when the runtime is the first one.
      if (child_runtime->GetModuleIndex() == 0) {
        child_runtime->SetInput(child_input_index, data_in);
      } else {
        if (!ForwardData(&forward_queue_map, child_runtime, child_input_index, data_in)) {
          return;
        }
      }
    }
    return;
  }
  /*!\brief Whether the output data is ready.*/
  bool DataIsReady(bool wait_data) {
    bool data_ready = true;
    for (auto queue_pair : input_queue_) {
      auto queue = queue_pair.second;
      if (queue->Empty()) {
        data_ready = false;
        break;
      }
    }
    if (!data_ready && wait_data) {
      // TODO(huajsj): Waitting the data ready message.
    }
    return data_ready;
  }
  /*!\brief Get the output data.*/
  bool GetOutput(Array<NDArray>* outputs, bool wait_data = false) {
    if (!DataIsReady(wait_data)) {
      return false;
    }
    for (auto queue_pair : input_queue_) {
      auto output_index = queue_pair.first;
      auto queue = queue_pair.second;
      QueueData data(const_cast<DLTensor*>(((*outputs)[output_index]).operator->()));
      if (!queue->Poll<QueueData>(&data)) {
        LOG(FATAL) << "There is no data in the data queue, it should not happen!";
      }
    }
    return true;
  }
  /*!\brief Initialized the data structures for pipeline.*/
  void InitializePipeline(InputConnectionConfig input_config,
                          const std::vector<std::shared_ptr<BackendRuntime>> runtimes) {
    input_config_ = input_config;
    runtimes_ = runtimes;
    for (auto child_runtime : runtimes) {
      int runtime_idx = child_runtime->GetModuleIndex();
      input_config.VisitConfig(
          [&](int input_index, int child_idx, std::string child_input_name) {
            auto child_input_index = child_runtime->GetInputIndex(child_input_name);
            if (child_input_index < 0) {
              LOG(FATAL) << "Can not find the input " << child_input_name << "in runtime "
                         << child_idx;
            }
            children_[input_index].push_back(std::make_pair(child_runtime, child_input_index));
            // Only create notify and queue for the runtime after the first runtime.
            if (runtime_idx != 0) {
              child_runtime->CreateParentsNotify(input_index, GLOBAL_MODULE_INDEX,
                                                 child_input_index);
              // Creating the pipeline forwarding queue.
              this->CreateForwardingQueue(input_index, child_runtime, child_input_index);
            }
          },
          runtime_idx);
    }
  }

 private:
  std::vector<std::shared_ptr<BackendRuntime>> runtimes_;
  InputConnectionConfig input_config_;
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
