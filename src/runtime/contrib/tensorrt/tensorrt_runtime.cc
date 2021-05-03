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

#include "../../file_utils.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

#ifdef TVM_GRAPH_EXECUTOR_TENSORRT
#include "NvInfer.h"
#include "tensorrt_builder.h"
#endif

namespace tvm {
namespace runtime {
namespace contrib {

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

using namespace tvm::runtime::json;

class TensorRTRuntime : public JSONRuntimeBase {
 public:
  /*!
   * \brief The TensorRT runtime module. Deserialize the provided functions
   * on creation and store in the layer cache.
   *
   * \param symbol_name The name of the function.
   * \param graph_json serialized JSON representation of a sub-graph.
   * \param const_names The names of each constant in the sub-graph.
   */
  explicit TensorRTRuntime(const std::string& symbol_name, const std::string& graph_json,
                           const Array<String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names),
        use_implicit_batch_(true),
        max_workspace_size_(size_t(1) << 30) {}

  /*!
   * \brief The type key of the module.
   *
   * \return module type key.
   */
  const char* type_key() const override { return "tensorrt"; }

  /*!
   * \brief Initialize runtime. Create TensorRT layer from JSON
   * representation.
   *
   * \param consts The constant params from compiled model.
   */
  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    LoadGlobalAttributes();
    if (GetCachedEnginesFromDisk()) return;
    SetupConstants(consts);
  }

  void LoadGlobalAttributes() {
    // These settings are global to the entire subgraph. Codegen will add them as attributes to all
    // op nodes. Read from first one.
    for (size_t i = 0; i < nodes_.size(); ++i) {
      if (nodes_[i].HasAttr("use_implicit_batch") && nodes_[i].HasAttr("max_workspace_size")) {
        use_implicit_batch_ =
            std::stoi(nodes_[i].GetAttr<std::vector<std::string>>("use_implicit_batch")[0]);
        // Allow max_workspace_size to be overridden at runtime.
        size_t runtime_max_workspace_size =
            dmlc::GetEnv("TVM_TENSORRT_MAX_WORKSPACE_SIZE", size_t(0));
        if (runtime_max_workspace_size != 0) {
          max_workspace_size_ = runtime_max_workspace_size;
        } else {
          max_workspace_size_ =
              std::stoul(nodes_[i].GetAttr<std::vector<std::string>>("max_workspace_size")[0]);
        }
        return;
      }
    }
  }

#ifdef TVM_GRAPH_EXECUTOR_TENSORRT
  /*! \brief Destroy engines and contexts. */
  ~TensorRTRuntime() {
    for (auto& it : trt_engine_cache_) {
      it.second.context->destroy();
      it.second.engine->destroy();
    }
  }

  /*! \brief Run inference using built engine. */
  void Run() override {
    BuildEngine();
    batch_size_ = data_entry_[input_var_eid_[0]]->shape[0];
    if (batch_size_ == 0) return;
    auto& engine_and_context = trt_engine_cache_.at(std::make_pair(symbol_name_, batch_size_));
    auto engine = engine_and_context.engine;
    auto context = engine_and_context.context;
    auto& device_buffers = engine_and_context.device_buffers;
    std::vector<void*> bindings(engine->getNbBindings(), nullptr);
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      if (nodes_[nid].GetOpType() == "input") {
        for (size_t j = 0; j < nodes_[nid].GetOpShape().size(); ++j) {
          uint32_t eid = EntryID(nid, j);
          const std::string name = nodes_[nid].GetOpName() + "_" + std::to_string(j);
          int binding_index = engine->getBindingIndex(name.c_str());
          ICHECK_NE(binding_index, -1);
          if (data_entry_[eid]->device.device_type == kDLGPU) {
            bindings[binding_index] = data_entry_[eid]->data;
          } else {
            device_buffers[binding_index].CopyFrom(data_entry_[eid]);
            bindings[binding_index] = device_buffers[binding_index]->data;
          }
        }
      }
    }

    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = EntryID(outputs_[i]);
      const std::string& name = engine_and_context.outputs[i];
      int binding_index = engine->getBindingIndex(name.c_str());
      ICHECK_NE(binding_index, -1);
      if (data_entry_[eid]->device.device_type == kDLGPU) {
        bindings[binding_index] = data_entry_[eid]->data;
      } else {
        bindings[binding_index] = device_buffers[binding_index]->data;
      }
    }

#if TRT_VERSION_GE(6, 0, 1)
    if (use_implicit_batch_) {
      ICHECK(context->execute(batch_size_, bindings.data())) << "Running TensorRT failed.";
    } else {
      ICHECK(context->executeV2(bindings.data())) << "Running TensorRT failed.";
    }
#else
    ICHECK(context->execute(batch_size_, bindings.data())) << "Running TensorRT failed.";
#endif

    // Copy outputs from GPU buffers if needed.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = EntryID(outputs_[i]);
      const std::string& name = engine_and_context.outputs[i];
      int binding_index = engine->getBindingIndex(name.c_str());
      ICHECK_NE(binding_index, -1);
      if (data_entry_[eid]->device.device_type != kDLGPU) {
        device_buffers[binding_index].CopyTo(const_cast<DLTensor*>(data_entry_[eid]));
      }
    }
  }

 private:
  /*!
   * \brief Build TensorRT engine from JSON representation and cache it. If engine is already built,
   * do nothing.
   */
  void BuildEngine() {
    batch_size_ =
        data_entry_[input_var_eid_[0]]->ndim == 0 ? 1 : data_entry_[input_var_eid_[0]]->shape[0];
    if (trt_engine_cache_.count(std::make_pair(symbol_name_, batch_size_))) return;
    DLOG(INFO) << "Building new TensorRT engine for subgraph " << symbol_name_
               << " with batch size " << batch_size_;
    const bool use_fp16 = dmlc::GetEnv("TVM_TENSORRT_USE_FP16", false);
    TensorRTBuilder builder(&logger_, data_entry_, max_workspace_size_, use_implicit_batch_,
                            use_fp16, batch_size_);

    // Add inputs and constants.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      const auto& node = nodes_[nid];
      std::string name = node.GetOpName();
      if (node.GetOpType() == "input") {
        builder.AddInput(nid, EntryID(nid, 0), node);
      } else {
        ICHECK_EQ(node.GetOpType(), "const");
        uint32_t eid = EntryID(nid, 0);
        builder.AddConstant(nid, data_entry_[eid]);
      }
    }

    // Add layers.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() != "kernel") continue;
      builder.AddLayer(nid, node);
    }

    // Add outputs.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      builder.AddOutput(outputs_[i], EntryID(outputs_[i]));
    }

    // Build engine.
    trt_engine_cache_[std::make_pair(symbol_name_, batch_size_)] = builder.BuildEngine();
    DLOG(INFO) << "Finished building TensorRT engine for subgraph " << symbol_name_
               << " with batch size " << batch_size_;
    CacheEngineToDisk();
  }

  /*! \brief If TVM_TENSORRT_CACHE_DIR is set, will check that directory for
   * already built TRT engines and load into trt_engine_cache_ so they don't
   * have to be built at first inference.
   */
  bool GetCachedEnginesFromDisk() {
    std::string cache_dir = dmlc::GetEnv("TVM_TENSORRT_CACHE_DIR", std::string(""));
    if (cache_dir.empty()) return false;
    std::string key = GetSubgraphKey();
    std::string path = cache_dir + "/" + key + ".plan";
    // Check if engine is in the cache.
    std::ifstream infile(path, std::ios::binary);
    if (!infile.good()) return false;
    DLOG(INFO) << "Loading cached TensorRT engine from " << path;
    infile.close();
    std::string serialized_engine;
    LoadBinaryFromFile(path, &serialized_engine);
    // Deserialize engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger_);
    TensorRTEngineAndContext engine_and_context;
    engine_and_context.engine =
        runtime->deserializeCudaEngine(&serialized_engine[0], serialized_engine.size(), nullptr);
    engine_and_context.context = engine_and_context.engine->createExecutionContext();
    // Load metadata
    std::string meta_path = cache_dir + "/" + key + ".meta";
    std::string serialized_meta;
    LoadBinaryFromFile(meta_path, &serialized_meta);
    std::istringstream is(serialized_meta);
    dmlc::JSONReader reader(&is);
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("inputs", &engine_and_context.inputs);
    helper.DeclareField("outputs", &engine_and_context.outputs);
    helper.ReadAllFields(&reader);
    const int batch_size = 1;
    trt_engine_cache_[std::make_pair(symbol_name_, batch_size)] = engine_and_context;
    return true;
  }

  /*! \brief If TVM_TENSORRT_CACHE_DIR is set, will save the engine to that
   * directory so it can be loaded later.
   */
  void CacheEngineToDisk() {
    batch_size_ = data_entry_[input_var_eid_[0]]->shape[0];
    std::string cache_dir = dmlc::GetEnv("TVM_TENSORRT_CACHE_DIR", std::string(""));
    if (cache_dir.empty()) return;
    std::string key = GetSubgraphKey();
    std::string path = cache_dir + "/" + key + ".plan";
    DLOG(INFO) << "Caching TensorRT engine to " << path;
    // Serialize engine to disk
    nvinfer1::IHostMemory* serialized_engine =
        trt_engine_cache_[std::make_pair(symbol_name_, batch_size_)].engine->serialize();
    SaveBinaryToFile(path, std::string(static_cast<const char*>(serialized_engine->data()),
                                       serialized_engine->size()));
    serialized_engine->destroy();
    // Serialize metadata
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    writer.BeginObject();
    writer.WriteObjectKeyValue("inputs",
                               trt_engine_cache_[std::make_pair(symbol_name_, batch_size_)].inputs);
    writer.WriteObjectKeyValue(
        "outputs", trt_engine_cache_[std::make_pair(symbol_name_, batch_size_)].outputs);
    writer.EndObject();
    std::string meta_path = cache_dir + "/" + key + ".meta";
    SaveBinaryToFile(meta_path, os.str());
  }

  std::string GetSubgraphKey() {
    // Using this key will only allow a single model per TVM_TENSORRT_CACHE_DIR directory. We could
    // instead use a hash of graph_json and all weights to allow many models in the same directory,
    // but the cost of computing the hash is high.
    return symbol_name_ + (dmlc::GetEnv("TVM_TENSORRT_USE_FP16", false) ? "_fp16" : "_fp32");
  }

  /*! \brief Get the batch size when in implicit_batch mode. */
  int GetBatchSize() {
    if (!use_implicit_batch_) return -1;
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      if (nodes_[nid].GetOpType() == "input") {
        // Get batch size from first input.
        return nodes_[nid].GetOpShape()[0][0];
      }
    }
    return -1;
  }

  /*! \brief Map of function name to TRT engine if built already. */
  std::unordered_map<std::pair<std::string, int>, TensorRTEngineAndContext, PairHash>
      trt_engine_cache_;

  /*! \brief TensorRT logger. */
  TensorRTLogger logger_;

  /*! \brief Batch size that the engine is optimized for. */
  int batch_size_;

#else
  void Run() override {
    LOG(FATAL) << "TensorRT runtime is not enabled. "
               << "Please build with USE_TENSORRT_RUNTIME.";
  }

  void BuildEngine() {
    LOG(WARNING) << "TensorRT runtime is not enabled. "
                 << "Please build with USE_TENSORRT_RUNTIME.";
  }

  bool GetCachedEnginesFromDisk() { return false; }

  void CacheEngineToDisk() {}
#endif

  bool use_implicit_batch_;

  size_t max_workspace_size_;
};

runtime::Module TensorRTRuntimeCreate(const String& symbol_name, const String& graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<TensorRTRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.tensorrt_runtime_create").set_body_typed(TensorRTRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_tensorrt")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<TensorRTRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
