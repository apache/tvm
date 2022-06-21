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
#include <dlpack/dlpack.h>
#include <dmlc/memory_io.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#include <tvm/target/target.h>

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "../base64.h"

namespace tvm {
namespace contrib {

struct ThreadLocalStore {
  tvm::runtime::Module mod;
  static ThreadLocalStore* ThreadLocal() {
    thread_local ThreadLocalStore tls;
    return &tls;
  }
};

class TVMScriptRuntimeClass : public torch::jit::CustomClassHolder {
 public:
  TVMScriptRuntimeClass() { mod_ = ThreadLocalStore::ThreadLocal()->mod; }

  void forward(const c10::List<at::Tensor>& inputs) {
    int input_length = inputs.size();

    std::vector<DLManagedTensor*> tensors;

    for (int i = 0; i < input_length; ++i) tensors.push_back(toDLPack(inputs[i]));

    tvm::runtime::PackedFunc run = mod_.GetFunction("__tvm_main__");

    std::vector<TVMValue> tvm_values(input_length);
    std::vector<int> tvm_type_codes(input_length);
    tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
    for (int k = 0; k < input_length; ++k) {
      setter(k, &tensors[k]->dl_tensor);
    }

    run.CallPacked(tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), input_length),
                   nullptr);

    for (int k = 0; k < input_length; ++k) {
      tensors[k]->deleter(tensors[k]);
    }
  }

 private:
  tvm::runtime::Module mod_;
};

tvm::Device getDevice(const at::Tensor& tensor) {
  tvm::Device dev;
  dev.device_id = tensor.get_device();
  switch (tensor.device().type()) {
    case at::DeviceType::CPU:
      dev.device_type = DLDeviceType::kDLCPU;
      if (dev.device_id == -1) {
        dev.device_id = 1;
      }
      break;
    case at::DeviceType::CUDA:
      dev.device_type = DLDeviceType::kDLCUDA;
      break;
    default:
      TORCH_CHECK(false, "PyTorch TVM integration doesn't support device " + tensor.device().str());
  }
  return dev;
}

class GraphExecutorFactoryWrapper : public torch::jit::CustomClassHolder {
 public:
  GraphExecutorFactoryWrapper(tvm::runtime::Module executor_factory)
      : executor_factory_(executor_factory) {}

  GraphExecutorFactoryWrapper()
      : GraphExecutorFactoryWrapper(ThreadLocalStore::ThreadLocal()->mod) {}

  c10::List<at::Tensor> forward(const c10::List<at::Tensor>& inputs) {
    int input_length = inputs.size();

    if (!executor_.defined()) {
      TORCH_CHECK(input_length > 0, "Receive empty list of input tensors");
      DLDevice input_device = getDevice(inputs.get(0));

      auto tmp = executor_factory_.GetFunction("default");

      executor_ = tmp(input_device);
    }

    std::vector<DLManagedTensor*> tensors;

    for (int i = 0; i < input_length; ++i) tensors.push_back(toDLPack(inputs[i]));

    tvm::runtime::PackedFunc run = executor_.GetFunction("run");
    tvm::runtime::PackedFunc set_input = executor_.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = executor_.GetFunction("get_output");
    tvm::runtime::PackedFunc get_num_outputs = executor_.GetFunction("get_num_outputs");

    for (int k = 0; k < input_length; ++k) {
      set_input(k, &tensors[k]->dl_tensor);
    }

    run();

    int64_t output_length = get_num_outputs();

    c10::List<at::Tensor> outputs;
    outputs.reserve(output_length);

    for (int k = 0; k < output_length; ++k) {
      tvm::runtime::NDArray results = get_output(k);
      at::Tensor atTensor = at::fromDLPack(results.ToDLPack());
      outputs.emplace_back(atTensor);
    }

    for (int k = 0; k < input_length; ++k) {
      tensors[k]->deleter(tensors[k]);
    }
    return outputs;
  }

  using SerializationType = std::string;  // executor factory stream

  SerializationType Serialize() {
    static const runtime::PackedFunc* f_to_str =
        runtime::Registry::Get("script_torch.save_to_base64");
    ICHECK(f_to_str) << "IndexError: Cannot find the packed function "
                        "`script_torch.save_to_tar` in the global registry";
    return (*f_to_str)(executor_factory_);
  }

  GraphExecutorFactoryWrapper(SerializationType state) {
    auto length = tvm::support::b64strlen(state);

    u_char bytes[length];
    memset(bytes, 0, sizeof(bytes));
    tvm::support::b64decode(state, bytes);

    const char* name = tmpnam(NULL);
    auto file_name = std::string(name) + ".so";
    auto pFile = fopen(file_name.c_str(), "wb");
    fwrite(bytes, sizeof(u_char), length, pFile);
    fclose(pFile);

    std::string load_f_name = "runtime.module.loadfile_so";
    const PackedFunc* f = runtime::Registry::Get(load_f_name);
    ICHECK(f != nullptr) << "Loader for `.so` files is not registered,"
                         << " resolved to (" << load_f_name << ") in the global registry."
                         << "Ensure that you have loaded the correct runtime code, and"
                         << "that you are on the correct hardware architecture.";

    executor_factory_ = (*f)(file_name, "");

    ICHECK(remove(file_name.c_str()) == 0)
        << "remove temporary file (" << file_name << ") unsuccessfully";
  }

 private:
  tvm::runtime::Module executor_factory_;
  tvm::runtime::Module executor_;
};

TVM_REGISTER_GLOBAL("tvmtorch.save_runtime_mod").set_body_typed([](tvm::runtime::Module mod) {
  ThreadLocalStore::ThreadLocal()->mod = mod;
});

TORCH_LIBRARY(tvm_torch, m) {
  m.class_<TVMScriptRuntimeClass>("TVMScriptRuntime")
      .def(torch::init<>())
      .def("forward", &TVMScriptRuntimeClass::forward);
}

TORCH_LIBRARY(tvm_tuning, m) {
  m.class_<GraphExecutorFactoryWrapper>("RelayRuntime")
      .def(torch::init<>())
      .def("forward", &GraphExecutorFactoryWrapper::forward)
      .def_pickle(
          [](const c10::intrusive_ptr<GraphExecutorFactoryWrapper>& self)
              -> GraphExecutorFactoryWrapper::SerializationType { return self->Serialize(); },
          [](GraphExecutorFactoryWrapper::SerializationType state) {
            return c10::make_intrusive<GraphExecutorFactoryWrapper>(state);
          });
}

}  // namespace contrib
}  // namespace tvm