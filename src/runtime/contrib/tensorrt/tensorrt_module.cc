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
#ifdef TVM_GRAPH_RUNTIME_TENSORRT
#include "tensorrt_builder.h"
#include "NvInfer.h"
#endif  // TVM_GRAPH_RUNTIME_TENSORRT

namespace tvm {
namespace runtime {

/*!
 * \brief Create a TensorRTModule.
 * \param serialized_subgraph Relay expr serialized with SaveJSON.
 * \return TensorRTModule created from subgraph.
 */
Module TensorRTModuleCreate(const std::string& serialized_subgraph);

/*! \brief A module for TensorRT runtime. */
class TensorRTModule : public runtime::ModuleNode {
 public:
  explicit TensorRTModule(const std::string& serialized_subgraph)
      : serialized_subgraph_(serialized_subgraph) {}

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
#if TVM_GRAPH_RUNTIME_TENSORRT
    // Generate an external packed function
    return PackedFunc([this, name](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      auto it = trt_engine_cache_.find(name);
      if (it == trt_engine_cache_.end()) {
        // Build new trt engine and place in cache.
        LOG(INFO) << "Building new TensorRT engine for subgraph " << name;
        auto expr = Downcast<relay::Expr>(LoadJSON(this->serialized_subgraph_));

        auto inputs = ConvertInputs(args);
        relay::contrib::TensorRTBuilder builder(inputs);
        auto engine_and_context = builder.BuildEngine(expr);
        LOG(INFO) << "Finished building engine";
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
    SaveBinaryToFile(file_name, serialized_subgraph_);
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(serialized_subgraph_);
  }

  static Module LoadFromFile(const std::string& path) {
    std::ifstream filep(path);
    filep.seekg(0, std::ios::end);
    size_t size = filep.tellg();
    std::string serialized_subgraph(size, ' ');
    filep.seekg(0);
    filep.read(&serialized_subgraph[0], size);
    return TensorRTModuleCreate(serialized_subgraph);
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string serialized_subgraph;
    stream->Read(&serialized_subgraph);
    return TensorRTModuleCreate(serialized_subgraph);
  }

 private:
  /*! \brief Relay program serialized using SaveJSON */
  std::string serialized_subgraph_;

#if TVM_GRAPH_RUNTIME_TENSORRT
  /*! \brief Map of function name to TRT engine if built already. */
  std::unordered_map<std::string, TrtEngineAndContext> trt_engine_cache_;

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
    const size_t num_outputs = engine_and_context.network_outputs.size();
    CHECK_GT(inputs.size(), num_outputs);
    // TODO(trevmorr): Assumes output is at the end - is this true?
    for (size_t i = 0; i < inputs.size() - num_outputs; ++i) {
      auto it = engine_and_context.network_input_map.find(i);
      if (it != engine_and_context.network_input_map.end()) {
        DLTensor* arg = inputs[i];
        int binding_index = engine->getBindingIndex(it->second.c_str());
        CHECK_NE(binding_index, -1);
        if (!runtime::TypeMatch(arg->dtype, kDLFloat, 32)) {
          LOG(FATAL) << "Only float32 inputs are supported.";
        }
        bindings[binding_index] = reinterpret_cast<float*>(arg->data);
      }
    }
    // Set outputs.
    for (size_t i = 0; i < num_outputs; ++i) {
      const int index_in_inputs = inputs.size() - num_outputs + i;
      DLTensor* out_arg = inputs[index_in_inputs];
      int binding_index = engine->getBindingIndex(
          engine_and_context.network_outputs[i].c_str());
      CHECK_NE(binding_index, -1);
      bindings[binding_index] = reinterpret_cast<float*>(out_arg->data);
    }
    // Use batch size from first input.
    const int batch_size = inputs[0]->shape[0];
    CHECK(context->execute(batch_size, bindings.data()))
        << "Running TensorRT failed.";
    *rv = bindings[num_bindings - num_outputs];
  }
#endif  // TVM_GRAPH_RUNTIME_TENSORRT
};

Module TensorRTModuleCreate(const std::string& serialized_subgraph) {
  auto n = make_object<TensorRTModule>(serialized_subgraph);
  return Module(n);
}

TVM_REGISTER_GLOBAL("tvm.contrib.tensorrt.create")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = TensorRTModuleCreate(args[0]);
});

TVM_REGISTER_GLOBAL("module.loadfile_tensorrt")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = TensorRTModule::LoadFromFile(args[0]);
});

TVM_REGISTER_GLOBAL("module.loadbinary_tensorrt")
.set_body_typed(TensorRTModule::LoadFromBinary);

}  // namespace runtime
}  // namespace tvm
