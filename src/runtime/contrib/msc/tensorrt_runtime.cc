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
 * \file src/runtime/contrib/tensorrt/tensorrt_runtime.cc
 * \brief JSON runtime implementation for TensorRT.
 */

#include <dmlc/parameter.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../json/json_runtime.h"

#ifdef TVM_GRAPH_EXECUTOR_TENSORRT
#include "../../../runtime/cuda/cuda_common.h"
#include "../tensorrt/tensorrt_logger.h"
#include "../tensorrt/tensorrt_utils.h"
#endif

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;

#ifdef TVM_GRAPH_EXECUTOR_TENSORRT
using namespace nvinfer1;
#endif

class MSCTensorRTRuntime : public JSONRuntimeBase {
 public:
  /*!
   * \brief The MSC TensorRT runtime module. Deserialize the provided functions
   * on creation and store in the layer cache.
   *
   * \param symbol_name The name of the function.
   * \param graph_json serialized JSON representation of a sub-graph.
   * \param const_names The names of each constant in the sub-graph.
   */
  explicit MSCTensorRTRuntime(const std::string& symbol_name, const std::string& graph_json,
                              const Array<String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  ~MSCTensorRTRuntime() override {
    VLOG(1) << "Destroying MSC TensorRT runtime";
    DestroyEngine();
  }

  /*!
   * \brief The type key of the module.
   *
   * \return module type key.
   */
  const char* type_key() const final { return "msc_tensorrt"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  }

  /*!
   * \brief Initialize runtime.
   *
   * \param consts The constant params from compiled model.
   */
  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    LoadGlobalOptions();
    for (size_t nid = 0; nid < nodes_.size(); nid++) {
      for (size_t oid = 0; oid < nodes_[nid].GetNumOutput(); oid++) {
        const auto& t_name = nodes_[nid].GetOpName() + ":" + std::to_string(oid);
        tensor_ids_[t_name] = std::make_pair(nid, oid);
      }
    }
    LoadEngine(engine_file_);
  }

  void LoadGlobalOptions() {
    // These settings are global to the entire subgraph. Codegen will add them as attributes to all
    // op nodes. Read from first one.
    for (size_t i = 0; i < nodes_.size(); ++i) {
      if (nodes_[i].HasAttr("msc_global_options_num")) {
        engine_file_ = nodes_[i].GetAttr<std::vector<std::string>>("msc_global_engine")[0];
        graph_name_ = nodes_[i].GetAttr<std::vector<std::string>>("msc_global_graph_name")[0];
        if (nodes_[i].HasAttr("msc_global_tool_tag")) {
          tool_tag_ = nodes_[i].GetAttr<std::vector<std::string>>("msc_global_tool_tag")[0];
        } else {
          tool_tag_ = "";
        }
      }
    }
  }

#ifdef TVM_GRAPH_EXECUTOR_TENSORRT
  void Run() override {
    SetInputOutputBinds();
    if (tool_tag_.size() > 0) {
      const auto* pf = runtime::Registry::Get("msc_tool.callback_step");
      ICHECK(pf != nullptr) << "Cannot find msc_tool.callback_step func.";
      Map<String, runtime::NDArray> input_datas;
      for (const auto& pair : input_bindings_) {
        const auto& tensor_name = engine_->getBindingName(pair.first);
        input_datas.Set(tensor_name, device_buffers_[pair.first]);
      }
      Map<String, Map<String, runtime::NDArray>> context;
      context.Set("datas", input_datas);
      (*pf)(context, "before_forward", graph_name_, tool_tag_);
    }
    auto tvm_stream = CUDAThreadEntry::ThreadLocal()->stream;
#if TRT_VERSION_GE(6, 0, 1)
    ICHECK(context_->enqueueV2(bindings_.data(), tvm_stream, nullptr))
        << "Running TensorRT failed.";
#else
    LOG_FATAL << "Only support tensorrt with version >=6.0.0";
#endif
    // Copy outputs from GPU buffers if needed.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto nid = outputs_[i].id_;
      uint32_t eid = EntryID(outputs_[i]);
      const auto& name = nodes_[nid].GetOpName() + ":" + std::to_string(outputs_[i].index_);
      int binding_index = engine_->getBindingIndex(name.c_str());
      ICHECK_NE(binding_index, -1);
      if (data_entry_[eid]->device.device_type != kDLCUDA || tool_tag_.size() > 0) {
        auto device_buffer = GetOrAllocateDeviceBuffer(eid, binding_index);
        device_buffer.CopyTo(const_cast<DLTensor*>(data_entry_[eid]));
      }
    }
    if (tool_tag_.size() > 0) {
      const auto* pf = runtime::Registry::Get("msc_tool.callback_step");
      ICHECK(pf != nullptr) << "Cannot find msc_tool.callback_step func.";
      Map<String, runtime::NDArray> output_datas;
      for (int bid = 0; bid < engine_->getNbBindings(); bid++) {
        if (input_bindings_.count(bid)) {
          continue;
        }
        const auto& tensor_name = engine_->getBindingName(bid);
        output_datas.Set(tensor_name, device_buffers_[bid]);
      }
      Map<String, Map<String, runtime::NDArray>> context;
      context.Set("datas", output_datas);
      (*pf)(context, "after_forward", graph_name_, tool_tag_);
    }
  }

  bool LoadEngine(const String& engine_file) {
    IRuntime* runtime = createInferRuntime(logger_);
    // build engine
    std::ifstream input(engine_file_, std::ifstream::binary);
    if (!input.is_open() || !input.good()) {
      LOG_ERROR << "Failed to open engine file " << engine_file_;
      return false;
    }
    std::vector<char> stream;
    size_t size = 0;
    input.seekg(0, input.end);
    size = input.tellg();
    input.seekg(0, input.beg);
    stream.resize(size);
    input.read(stream.data(), size);
    input.close();

#if TRT_VERSION_GE(8, 0, 0)
    engine_ = runtime->deserializeCudaEngine(stream.data(), size);
#else
    engine_ = runtime->deserializeCudaEngine(stream.data(), size, nullptr);
#endif
    if (!engine_) {
      LOG_ERROR << "Failed to load engine";
      return false;
    }
    // create context
    context_ = engine_->createExecutionContext();
    if (!context_) {
      LOG_ERROR << "Failed to create context";
      return false;
    }
    // resize bindings
    size_t num_binding = static_cast<size_t>(engine_->getNbBindings());
    bindings_.resize(num_binding);
    binding_sizes_.resize(num_binding);
    for (size_t i = 0; i < num_binding; i++) {
      bindings_[i] = nullptr;
      binding_sizes_[i] = 0;
    }
    // destroy runtime
#if TRT_VERSION_GE(8, 0, 0)
    delete runtime;
#else
    runtime->destroy();
#endif
    return true;
  }

  void DestroyEngine() {
#if TRT_VERSION_GE(8, 0, 0)
    delete context_;
    delete engine_;
#else
    context_->destroy();
    engine_->destroy();
#endif
    engine_ = nullptr;
    context_ = nullptr;
  }

  void SetInputOutputBinds() {
    // Setup input bindings
    std::set<int> binded;
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      if (nodes_[nid].GetOpType() == "input") {
        for (size_t j = 0; j < nodes_[nid].GetOpShape().size(); ++j) {
          uint32_t eid = EntryID(nid, j);
          const auto& name = nodes_[nid].GetOpName() + ":" + std::to_string(j);
          int binding_index = engine_->getBindingIndex(name.c_str());
          ICHECK_NE(binding_index, -1);
#if TRT_VERSION_GE(6, 0, 1)
          std::vector<int64_t> shape(data_entry_[eid]->shape,
                                     data_entry_[eid]->shape + data_entry_[eid]->ndim);
          ICHECK(context_->setBindingDimensions(binding_index, VectorToTrtDims(shape)));
#endif
          if (data_entry_[eid]->device.device_type == kDLCUDA && tool_tag_.size() == 0) {
            bindings_[binding_index] = data_entry_[eid]->data;
          } else {
            auto device_buffer = GetOrAllocateDeviceBuffer(eid, binding_index);
            device_buffer.CopyFrom(data_entry_[eid]);
            bindings_[binding_index] = device_buffer->data;
          }
          auto dims = engine_->getBindingDimensions(binding_index);
          int num_elements = 1;
          for (int i = 0; i < dims.nbDims; ++i) num_elements *= dims.d[i];
          binding_sizes_[binding_index] = num_elements;
          input_bindings_[binding_index] = eid;
          binded.insert(binding_index);
        }
      }
    }
    // Setup output bindings.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto nid = outputs_[i].id_;
      uint32_t eid = EntryID(outputs_[i]);
      const auto& name = nodes_[nid].GetOpName() + ":" + std::to_string(outputs_[i].index_);
      int binding_index = engine_->getBindingIndex(name.c_str());
      ICHECK_NE(binding_index, -1);
      if (data_entry_[eid]->device.device_type == kDLCUDA && tool_tag_.size() == 0) {
        bindings_[binding_index] = data_entry_[eid]->data;
      } else {
        auto device_buffer = GetOrAllocateDeviceBuffer(eid, binding_index);
        bindings_[binding_index] = device_buffer->data;
      }
      output_bindings_[binding_index] = eid;
      binded.insert(binding_index);
    }
    // Setup tool bindings
    for (int bid = 0; bid < engine_->getNbBindings(); bid++) {
      if (binded.count(bid)) {
        continue;
      }
      if (!device_buffers_.count(bid)) {
        const auto& tensor_name = engine_->getBindingName(bid);
        ICHECK(tensor_ids_.count(tensor_name)) << "Can not find tensor_name " << tensor_name;
        const auto& pair = tensor_ids_[tensor_name];
        auto shape = nodes_[pair.first].GetOpShape()[pair.second];
        auto dtype = nodes_[pair.first].GetOpDataType()[pair.second];
        device_buffers_[bid] = runtime::NDArray::Empty(shape, dtype, {kDLCUDA, 0});
      }
      bindings_[bid] = device_buffers_[bid]->data;
      binded.insert(bid);
    }
  }

  NDArray GetOrAllocateDeviceBuffer(int entry_id, int binding_index) {
    std::vector<int64_t> shape(data_entry_[entry_id]->shape,
                               data_entry_[entry_id]->shape + data_entry_[entry_id]->ndim);
    if (device_buffers_.count(binding_index)) {
      // Buffer is already initialized.
      if (shape[0] > device_buffers_[binding_index]->shape[0]) {
        // Buffer is too small. Need to allocate bigger buffer.
        device_buffers_[binding_index] =
            runtime::NDArray::Empty(shape, data_entry_[entry_id]->dtype, {kDLCUDA, 0});
      } else if (shape[0] < device_buffers_[binding_index]->shape[0]) {
        // Buffer is too large. Create view.
        return device_buffers_[binding_index].CreateView(shape, data_entry_[entry_id]->dtype);
      }
    } else {
      // Buffer not initialized yet.
      device_buffers_[binding_index] =
          runtime::NDArray::Empty(shape, data_entry_[entry_id]->dtype, {kDLCUDA, 0});
    }
    return device_buffers_.at(binding_index);
  }

#else   // TVM_GRAPH_EXECUTOR_TENSORRT
  void Run() override {
    LOG(FATAL) << "TensorRT runtime is not enabled. "
               << "Please build with USE_TENSORRT_RUNTIME.";
  }

  bool LoadEngine(const String& engine_file) { return false; }

  void DestroyEngine() {}
#endif  // TVM_GRAPH_EXECUTOR_TENSORRT

 private:
  String engine_file_;
  String tool_tag_;
  String graph_name_;
  std::unordered_map<std::string, std::pair<size_t, size_t>> tensor_ids_;
#ifdef TVM_GRAPH_EXECUTOR_TENSORRT
  TensorRTLogger logger_;
  ICudaEngine* engine_{nullptr};
  IExecutionContext* context_{nullptr};
  std::unordered_map<int, uint32_t> input_bindings_;
  std::unordered_map<int, uint32_t> output_bindings_;
  std::vector<void*> bindings_;
  std::vector<size_t> binding_sizes_;
  std::unordered_map<int, NDArray> device_buffers_;
#endif
};

runtime::Module MSCTensorRTRuntimeCreate(const String& symbol_name, const String& graph_json,
                                         const Array<String>& const_names) {
  auto n = make_object<MSCTensorRTRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.msc_tensorrt_runtime_create").set_body_typed(MSCTensorRTRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_msc_tensorrt")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<MSCTensorRTRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
