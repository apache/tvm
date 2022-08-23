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
#include <ATen/DLConvertor.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <iostream>

#include "runtime_bridge.h"

namespace tvm {
namespace contrib {

/*
 * Convert Torch tensor to DLPack extended tensor.
 * The boolean Torch tensor will convert to DLtensor with `is_bool=True` flag.
 * @param src Torch tensor
 * @return DLPack extended tensor
 */
DLPackTensorExt ToDLPackExt(const at::Tensor& src) {
  if (!src.is_contiguous()) {
    return ToDLPackExt(src.contiguous());
  }
  DLPackTensorExt ret;
  if (src.dtype().isScalarType(torch::kBool)) {
    auto temp = src.toType(torch::kUInt8);
    ret.dl_managed_tensor = at::toDLPack(temp);
    ret.is_bool = true;
  } else {
    ret.dl_managed_tensor = at::toDLPack(src);
    ret.is_bool = false;
  }

  return ret;
}

/*
 * Convert DLPack extended tensor to Torch tensor.
 * @param src DLPack extended tensor
 * @return Torch tensor
 */
at::Tensor FromDLPackExt(const DLPackTensorExt& src) {
  if (src.is_bool) {
    return at::fromDLPack(src.dl_managed_tensor).toType(torch::kBool);
  } else {
    return at::fromDLPack(src.dl_managed_tensor);
  }
}

/**
 * @brief A Torch's module which wraps TVM's OperatorModule Class.
 * The basic forward function calling TVM's runtime is provided.
 * The TVM module can be serialized/deserialized as a Torch module.
 */
class OperatorModuleWrapper : public torch::jit::CustomClassHolder {
 public:
  OperatorModuleWrapper() { runtime_module_ = tvm_contrib_torch_get_last_saved_runtime_module(); }
  ~OperatorModuleWrapper() { tvm_contrib_torch_free_runtime_module(runtime_module_); }

  void forward(const c10::List<at::Tensor>& inputs) {
    int input_length = inputs.size();

    std::vector<DLPackTensorExt> tensors;

    // Torch tensor supports boolean type while DLpack does not,
    // we convert Torch tensor to an extension of DLPack tensor
    for (int i = 0; i < input_length; ++i) tensors.push_back(ToDLPackExt(inputs[i]));
    tvm_contrib_torch_operator_module_forward(this->runtime_module_, tensors.data(),
                                              tensors.size());

    for (int k = 0; k < input_length; ++k) {
      if (tvm_contrib_torch_tensor_ability_of_zero_copy(&tensors[k])) {
        // We need to free memory manually
        tensors[k].dl_managed_tensor->deleter(tensors[k].dl_managed_tensor);
      } else {
        // Ownership transferred
        inputs[k].copy_(FromDLPackExt(tensors[k]));
      }
    }
  }

  std::string Serialize() {
    auto encoding = tvm_contrib_torch_encode(runtime_module_);
    auto ret = std::string(encoding);
    tvm_contrib_torch_free_encoding(encoding);
    return ret;
  }

  explicit OperatorModuleWrapper(std::string state) {
    runtime_module_ = tvm_contrib_torch_decode(state.c_str());
  }

 private:
  /*
   * TVM runtime module wrapper
   */
  TVMContribTorchRuntimeModule* runtime_module_;
};

/**
 * @brief A Torch's module which wraps TVM's GraphExecutorFactory Class.
 * The basic forward function calling TVM's runtime is provided.
 * The TVM module can be serialized/deserialized as a Torch module.
 */
class GraphExecutorFactoryWrapper : public torch::jit::CustomClassHolder {
 public:
  explicit GraphExecutorFactoryWrapper(TVMContribTorchRuntimeModule* executor_factory)
      : executor_factory_(executor_factory), executor_factory_runtime_(nullptr) {}

  ~GraphExecutorFactoryWrapper() {
    tvm_contrib_torch_free_runtime_module(executor_factory_);
    tvm_contrib_torch_free_runtime_module(executor_factory_runtime_);
  }

  GraphExecutorFactoryWrapper()
      : GraphExecutorFactoryWrapper(tvm_contrib_torch_get_last_saved_runtime_module()) {}

  std::string Serialize() {
    auto encoding = tvm_contrib_torch_encode(executor_factory_);
    auto ret = std::string(encoding);
    tvm_contrib_torch_free_encoding(encoding);
    return ret;
  }

  explicit GraphExecutorFactoryWrapper(std::string state) {
    executor_factory_ = tvm_contrib_torch_decode(state.c_str());
    executor_factory_runtime_ = nullptr;
  }

  c10::List<at::Tensor> forward(const c10::List<at::Tensor>& inputs) {
    int input_length = inputs.size();

    TORCH_CHECK(input_length > 0, "Receive empty list of input tensors");

    std::vector<DLPackTensorExt> tensors;

    // Torch tensor supports boolean type while DLpack does not,
    // we convert Torch tensor to an extension of DLPack tensor
    for (int i = 0; i < input_length; ++i) tensors.push_back(ToDLPackExt(inputs[i]));

    DLPackTensorExt* outputs;
    if (executor_factory_runtime_ == nullptr) {
      executor_factory_runtime_ = tvm_contrib_torch_create_graph_runtime_module(
          this->executor_factory_, tensors[0].dl_managed_tensor);
    }
    auto num_outputs = tvm_contrib_torch_graph_executor_module_forward(
        executor_factory_runtime_, tensors.data(), tensors.size(), &outputs);

    c10::List<at::Tensor> ret;
    ret.reserve(num_outputs);

    for (size_t k = 0; k < num_outputs; ++k) {
      at::Tensor atTensor = FromDLPackExt(outputs[k]);
      ret.emplace_back(atTensor);
    }

    for (int k = 0; k < input_length; ++k) {
      tensors[k].dl_managed_tensor->deleter(tensors[k].dl_managed_tensor);
    }
    tvm_contrib_torch_free_dlpack_tensor_ext_array(outputs);

    return ret;
  }

 private:
  /*
   * TVM Graph Executor Factory module wrapper
   */
  TVMContribTorchRuntimeModule* executor_factory_;

  /*
   * TVM runtime module wrapper
   */
  TVMContribTorchRuntimeModule* executor_factory_runtime_;
};

TORCH_LIBRARY(tvm_torch, m) {
  m.class_<OperatorModuleWrapper>("OperatorModuleWrapper")
      .def(torch::init<>())
      .def("forward", &OperatorModuleWrapper::forward)
      .def_pickle(
          [](const c10::intrusive_ptr<OperatorModuleWrapper>& self) -> std::string {
            return self->Serialize();
          },
          [](std::string state) { return c10::make_intrusive<OperatorModuleWrapper>(state); });
  m.class_<GraphExecutorFactoryWrapper>("GraphExecutorFactoryWrapper")
      .def(torch::init<>())
      .def("forward", &GraphExecutorFactoryWrapper::forward)
      .def_pickle(
          [](const c10::intrusive_ptr<GraphExecutorFactoryWrapper>& self) -> std::string {
            return self->Serialize();
          },
          [](std::string state) {
            return c10::make_intrusive<GraphExecutorFactoryWrapper>(state);
          });
}

}  // namespace contrib
}  // namespace tvm
