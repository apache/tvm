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

DLPackTensorExt toDLPackExt(const at::Tensor& src) {
  if (!src.is_contiguous()) {
    return toDLPackExt(src.contiguous());
  }

  if (src.dtype().isScalarType(torch::kBool)) {
    auto temp = src.toType(torch::kUInt8);
    return {.dl_managed_tensor = at::toDLPack(temp), .is_bool = true};
  }

  return {.dl_managed_tensor = at::toDLPack(src), .is_bool = false};
}

at::Tensor fromDLPackExt(const DLPackTensorExt& src) {
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
  OperatorModuleWrapper() { runtime_module = tvm_contrib_torch_get_last_saved_runtime_module(); }

  void forward(const c10::List<at::Tensor>& inputs) {
    int input_length = inputs.size();

    std::vector<DLPackTensorExt> tensors;

    for (int i = 0; i < input_length; ++i) tensors.push_back(toDLPackExt(inputs[i]));
    tvm_contrib_torch_operator_module_forward(
        this->runtime_module, static_cast<DLPackTensorExt*>(tensors.data()), tensors.size());

    for (int k = 0; k < input_length; ++k) {
      tensors[k].dl_managed_tensor->deleter(tensors[k].dl_managed_tensor);
    }
  }

  std::string Serialize() { return std::string(tvm_contrib_torch_encode(runtime_module)); }

  explicit OperatorModuleWrapper(std::string state) {
    runtime_module = tvm_contrib_torch_decode(state.c_str());
  }

 private:
  TVMContribTorchRuntimeModule* runtime_module;
};

/**
 * @brief A Torch's module which wraps TVM's GraphExecutorFactory Class.
 * The basic forward function calling TVM's runtime is provided.
 * The TVM module can be serialized/deserialized as a Torch module.
 */
class GraphExecutorFactoryWrapper : public torch::jit::CustomClassHolder {
 public:
  explicit GraphExecutorFactoryWrapper(TVMContribTorchRuntimeModule* executor_factory)
      : executor_factory_(executor_factory) {}

  GraphExecutorFactoryWrapper()
      : GraphExecutorFactoryWrapper(tvm_contrib_torch_get_last_saved_runtime_module()) {}
  std::string Serialize() { return tvm_contrib_torch_encode(executor_factory_); }

  explicit GraphExecutorFactoryWrapper(std::string state) {
    executor_factory_ = tvm_contrib_torch_decode(state.c_str());
  }

  c10::List<at::Tensor> forward(const c10::List<at::Tensor>& inputs) {
    int input_length = inputs.size();

    TORCH_CHECK(input_length > 0, "Receive empty list of input tensors");

    std::vector<DLPackTensorExt> tensors;

    for (int i = 0; i < input_length; ++i) tensors.push_back(toDLPackExt(inputs[i]));

    auto outputs = new DLPackTensorExt*;

    auto num_outputs = tvm_contrib_torch_graph_executor_module_forward(
        executor_factory_, static_cast<DLPackTensorExt*>(tensors.data()), tensors.size(), outputs);

    c10::List<at::Tensor> ret;
    ret.reserve(num_outputs);

    for (int k = 0; k < num_outputs; ++k) {
      at::Tensor atTensor = fromDLPackExt((*outputs)[k]);
      ret.emplace_back(atTensor);
    }

    for (int k = 0; k < input_length; ++k) {
      tensors[k].dl_managed_tensor->deleter(tensors[k].dl_managed_tensor);
    }

    delete outputs;

    return ret;
  }

 private:
  TVMContribTorchRuntimeModule* executor_factory_;
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
