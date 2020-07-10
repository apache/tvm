/* * Licensed to the Apache Software Foundation (ASF) under one
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
 * \file runtime/contrib/tensorrt/tensorrt_module.cc
 * \brief TensorRTModule is the runtime module for tensorrt backend.
 */

#include <stdlib.h>
#include <tvm/node/serialization.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../../file_util.h"
#include "tensorrt_module.h"
#ifdef TVM_GRAPH_RUNTIME_TENSORRT
#include "NvInfer.h"
#include "tensorrt_builder.h"
#endif  // TVM_GRAPH_RUNTIME_TENSORRT

namespace tvm {
namespace runtime {

/*! \brief A module for TensorRT runtime. */
class TensorRTModule : public runtime::ModuleNode {
 public:
  explicit TensorRTModule(
      const std::unordered_map<std::string, std::string>& serialized_subgraphs)
      : serialized_subgraphs_(serialized_subgraphs) {
    max_workspace_size_ = dmlc::GetEnv("TVM_TENSORRT_MAX_WORKSPACE_SIZE", size_t(1) << 31);
    use_implicit_batch_ = dmlc::GetEnv("TVM_TENSORRT_USE_IMPLICIT_BATCH", true);
#if TVM_GRAPH_RUNTIME_TENSORRT
    GetCachedEnginesFromDisk();
#endif
  }

  ~TensorRTModule() {
#if TVM_GRAPH_RUNTIME_TENSORRT
    for (auto& it : trt_engine_cache_) {
      it.second.context->destroy();
      it.second.engine->destroy();
    }
#endif  // TVM_GRAPH_RUNTIME_TENSORRT
  }

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    // Returning nullptr tells TVM that the function is not in this module, so
    // it can look for the correct one.
    auto it_subgraph = serialized_subgraphs_.find(name);
    if (it_subgraph == serialized_subgraphs_.end()) {
      return PackedFunc(nullptr);
    }
#if TVM_GRAPH_RUNTIME_TENSORRT
    // Generate an external packed function
    return PackedFunc([this, name](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      auto it = trt_engine_cache_.find(name);
      if (it == trt_engine_cache_.end()) {
        // Build new trt engine and place in cache.
        LOG(INFO) << "Building new TensorRT engine for subgraph " << name;
        auto func = Downcast<relay::Function>(
            LoadJSON(this->serialized_subgraphs_[name]));
        auto inputs = ConvertInputs(args);
        std::string key = GetSubgraphKey(serialized_subgraphs_[name]);
        this->serialized_subgraphs_[name].clear();
        relay::contrib::TensorRTBuilder builder(&logger_, inputs, max_workspace_size_,
                                                use_implicit_batch_);
        auto engine_and_context = builder.BuildEngine(func);
        CacheEngineToDisk(key, engine_and_context);
        LOG(INFO) << "Finished building TensorRT engine for subgraph " << name;
        this->trt_engine_cache_[name] = engine_and_context;
        this->ExecuteEngine(engine_and_context, args, rv);
      } else {
        this->ExecuteEngine(it->second, args, rv);
      }
    });
#else
    LOG(FATAL) << "TVM was not built with TensorRT runtime enabled. Build "
               << "with USE_TENSORRT=ON.";
    return PackedFunc();
#endif  // TVM_GRAPH_RUNTIME_TENSORRT
  }

  const char* type_key() const { return "tensorrt"; }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    CHECK_EQ(fmt, type_key()) << "Can only save to format=" << type_key();
    SaveBinaryToFile(file_name, SerializeModuleToString());
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(SerializeModuleToString());
  }

  static Module LoadFromFile(const std::string& path) {
    std::ifstream filep(path);
    filep.seekg(0, std::ios::end);
    size_t size = filep.tellg();
    std::string serialized_module(size, ' ');
    filep.seekg(0);
    filep.read(&serialized_module[0], size);
    return CreateModuleFromString(serialized_module);
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string serialized_module;
    stream->Read(&serialized_module);
    return CreateModuleFromString(serialized_module);
  }

 private:
  /*! \brief Relay program serialized using SaveJSON */
  std::unordered_map<std::string, std::string> serialized_subgraphs_;

  /*! \brief Max workspace size for TensorRT */
  size_t max_workspace_size_;

  /*! \brief Whether to use implicit batch mode. */
  bool use_implicit_batch_;

#if TVM_GRAPH_RUNTIME_TENSORRT
  /*! \brief Map of function name to TRT engine if built already. */
  std::unordered_map<std::string, TrtEngineAndContext> trt_engine_cache_;

  /*! \brief TensorRT object used to log warnings and errors. */
  TensorRTLogger logger_;

  /*!
   * \brief Convert TVMArgs to make compatible with VM or graph runtime.
   * \param args Inputs to the PackedFunc.
   * \return Inputs converted to vector of DLTensor*
   */
  std::vector<DLTensor*> ConvertInputs(tvm::TVMArgs args) {
    std::vector<DLTensor*> inputs(args.size(), nullptr);
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i].type_code() == kTVMNDArrayHandle) {
        // Relay Debug/VM uses NDArray
        runtime::NDArray array = args[i];
        inputs[i] = const_cast<DLTensor*>(array.operator->());
      } else if (args[i].type_code() == kTVMDLTensorHandle) {
        // Graph runtime uses DLTensors
        inputs[i] = args[i];
      } else {
        LOG(FATAL) << "Invalid TVMArgs type.";
      }
    }
    return inputs;
  }

  /*!
   * \brief Perform inference using TensorRT.
   * \param engine_and_context TRT engine from TrtBuilder::BuildEngine()
   * \param args Inputs to the PackedFunc.
   * \param rv Return value pointer for the PackedFunc.
   * \return Inputs converted to vector of DLTensor*
   */
  void ExecuteEngine(const TrtEngineAndContext& engine_and_context,
                     tvm::TVMArgs args, tvm::TVMRetValue* rv) {
    auto engine = engine_and_context.engine;
    auto context = engine_and_context.context;
    const int num_bindings = engine->getNbBindings();
    std::vector<void*> bindings(num_bindings, nullptr);
    // Set inputs.
    auto inputs = ConvertInputs(args);
    const size_t num_outputs = engine_and_context.outputs.size();
    CHECK_GT(inputs.size(), num_outputs);
    for (size_t i = 0; i < engine_and_context.inputs.size(); ++i) {
      // If an input was baked into the engine, skip.
      if (engine_and_context.input_is_baked[i]) continue;
      DLTensor* arg = inputs[i];
      int binding_index =
          engine->getBindingIndex(engine_and_context.inputs[i].c_str());
      CHECK_NE(binding_index, -1);
      if (!runtime::TypeMatch(arg->dtype, kDLFloat, 32)) {
        LOG(FATAL) << "Only float32 inputs are supported.";
      }
      bindings[binding_index] = reinterpret_cast<float*>(arg->data);
#if TRT_VERSION_GE(6, 0, 1)
      // Set binding dimensions for INetworkV2 explicit batch mode engines.
      if (!use_implicit_batch_) {
        nvinfer1::Dims dims;
        dims.d[0] = 1;
        dims.nbDims = arg->ndim;
        for (int i = 0; i < arg->ndim; ++i) {
          dims.d[i] = arg->shape[i];
        }
        context->setBindingDimensions(binding_index, dims);
      }
#endif
    }
    // Set outputs.
    for (size_t i = 0; i < num_outputs; ++i) {
      const int index_in_inputs = inputs.size() - num_outputs + i;
      DLTensor* out_arg = inputs[index_in_inputs];
      int binding_index =
          engine->getBindingIndex(engine_and_context.outputs[i].c_str());
      CHECK_NE(binding_index, -1);
      bindings[binding_index] = reinterpret_cast<float*>(out_arg->data);
    }
#if TRT_VERSION_GE(6, 0, 1)
    if (use_implicit_batch_) {
      // Use batch size from first input.
      const int batch_size = inputs[0]->shape[0];
      CHECK(context->execute(batch_size, bindings.data())) << "Running TensorRT failed.";
    } else {
      CHECK(context->executeV2(bindings.data())) << "Running TensorRT failed.";
    }
#else
    // Use batch size from first input.
    const int batch_size = inputs[0]->shape[0];
    CHECK(context->execute(batch_size, bindings.data())) << "Running TensorRT failed.";
#endif
    *rv = bindings[num_bindings - num_outputs];
  }

  std::string GetSubgraphKey(const std::string& serialized_subgraph) {
    if (dmlc::GetEnv("TVM_TENSORRT_CACHE_DIR", std::string("")).empty()) return "";
    std::string key = std::to_string(std::hash<std::string>()(serialized_subgraph));
    if (dmlc::GetEnv("TVM_TENSORRT_USE_FP16", false)) {
      key += "_fp16";
    }
    return key;
  }

  /*! \brief If TVM_TENSORRT_CACHE_DIR is set, will check that directory for
   * already built TRT engines and load into trt_engine_cache_ so they don't
   * have to be built at first inference.
   */
  void GetCachedEnginesFromDisk() {
    std::string cache_dir = dmlc::GetEnv("TVM_TENSORRT_CACHE_DIR", std::string(""));
    if (cache_dir.empty()) return;
    for (auto it : serialized_subgraphs_) {
      std::string key = GetSubgraphKey(it.second);
      std::string path = cache_dir + "/" + key + ".plan";
      // Check if engine is in the cache.
      std::ifstream infile(path, std::ios::binary);
      if (!infile.good()) continue;
      LOG(INFO) << "Loading cached TensorRT engine from " << path;
      infile.close();
      std::string serialized_engine;
      LoadBinaryFromFile(path, &serialized_engine);
      // Deserialize engine
      nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger_);
      TrtEngineAndContext engine_and_context;
      engine_and_context.engine = runtime->deserializeCudaEngine(
          &serialized_engine[0], serialized_engine.size(), nullptr);;
      engine_and_context.context = engine_and_context.engine->createExecutionContext();
      // Load metadata
      std::string meta_path = cache_dir + "/" + key + ".meta";
      std::string serialized_meta;
      LoadBinaryFromFile(meta_path, &serialized_meta);
      std::istringstream is(serialized_meta);
      dmlc::JSONReader reader(&is);
      dmlc::JSONObjectReadHelper helper;
      helper.DeclareField("inputs", &engine_and_context.inputs);
      helper.DeclareField("input_is_baked", &engine_and_context.input_is_baked);
      helper.DeclareField("outputs", &engine_and_context.outputs);
      helper.ReadAllFields(&reader);
      trt_engine_cache_[it.first] = engine_and_context;
    }
  }

  /*! \brief If TVM_TENSORRT_CACHE_DIR is set, will save the engine to that
   * directory so it can be loaded later. A hash of the source relay function is
   * used as the key for the file name.
   * \param name Subgraph name
   * \param engine_and_context Engine to cache
   */
  void CacheEngineToDisk(const std::string& key, const TrtEngineAndContext& engine_and_context) {
    std::string cache_dir = dmlc::GetEnv("TVM_TENSORRT_CACHE_DIR", std::string(""));
    if (cache_dir.empty()) return;
    std::string path = cache_dir + "/" + key + ".plan";
    LOG(INFO) << "Caching TensorRT engine to " << path;
    // Serialize engine to disk
    nvinfer1::IHostMemory* serialized_engine = engine_and_context.engine->serialize();
    SaveBinaryToFile(path, std::string(static_cast<const char*>(serialized_engine->data()),
                                       serialized_engine->size()));
    serialized_engine->destroy();
    // Serialize metadata
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    writer.BeginObject();
    writer.WriteObjectKeyValue("inputs", engine_and_context.inputs);
    writer.WriteObjectKeyValue("input_is_baked", engine_and_context.input_is_baked);
    writer.WriteObjectKeyValue("outputs", engine_and_context.outputs);
    writer.EndObject();
    std::string meta_path = cache_dir + "/" + key + ".meta";
    SaveBinaryToFile(meta_path, os.str());
  }
#endif  // TVM_GRAPH_RUNTIME_TENSORRT

  /*! \brief Serialize this module to a string. To be used during codegen. */
  std::string SerializeModuleToString() {
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    writer.BeginObject();
    writer.WriteObjectKeyValue("subgraphs", serialized_subgraphs_);
    writer.WriteObjectKeyValue("max_workspace_size", max_workspace_size_);
    writer.WriteObjectKeyValue("use_implicit_batch", use_implicit_batch_);
    writer.EndObject();
    return os.str();
  }

  /*! \brief Load serialized module from string created by SerializeModuleToString. */
  static Module CreateModuleFromString(const std::string& str) {
    std::unordered_map<std::string, std::string> serialized_subgraphs;
    size_t max_workspace_size = 0;
    bool use_implicit_batch = true;
    std::istringstream is(str);
    dmlc::JSONReader reader(&is);
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("subgraphs", &serialized_subgraphs);
    helper.DeclareOptionalField("max_workspace_size", &max_workspace_size);
    helper.DeclareOptionalField("use_implicit_batch", &use_implicit_batch);
    helper.ReadAllFields(&reader);
    auto n = make_object<TensorRTModule>(serialized_subgraphs);
    // Use max_workspace_size from artifact if it is set and it is not overriden by env var.
    if (max_workspace_size != 0 && dmlc::GetEnv("TVM_TENSORRT_MAX_WORKSPACE_SIZE", 0) != 0) {
      n->max_workspace_size_ = max_workspace_size;
    }
    n->use_implicit_batch_ = use_implicit_batch;
    return Module(n);
  }
};

Module TensorRTModuleCreate(
    const std::unordered_map<std::string, std::string>& serialized_subgraphs) {
  auto n = make_object<TensorRTModule>(serialized_subgraphs);
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_tensorrt")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = TensorRTModule::LoadFromFile(args[0]);
});

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_tensorrt")
.set_body_typed(TensorRTModule::LoadFromBinary);

}  // namespace runtime
}  // namespace tvm
