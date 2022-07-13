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

#include "../../../runtime/graph_executor/graph_executor_factory.h"
#include "../base64.h"

namespace tvm {
namespace contrib {

/**
 * We pass the TVM module by TVM's FFI because Torch's FFI cannot recognize such TVM objects
 */
struct ThreadLocalStore {
  tvm::runtime::Module mod;
  static ThreadLocalStore* ThreadLocal() {
    thread_local ThreadLocalStore tls;
    return &tls;
  }
};

using SerializationType = std::string;  // base64 stream

SerializationType serialize(tvm::runtime::Module module) {
  static const runtime::PackedFunc* f_to_str =
      runtime::Registry::Get("script_torch.save_to_base64");
  ICHECK(f_to_str) << "IndexError: Cannot find the packed function "
                      "`script_torch.save_to_base64` in the global registry";
  return (*f_to_str)(module);
}

struct Deleter {  // deleter
  explicit Deleter(std::string file_name) { this->file_name = file_name; }
  void operator()(FILE* p) const {
    fclose(p);
    ICHECK(remove(file_name.c_str()) == 0)
        << "Failed to  remove temporary file (" << file_name << ")";
  }
  std::string file_name;
};

tvm::runtime::Module deserialize(SerializationType state) {
  auto length = tvm::support::b64strlen(state);

  std::vector<u_char> bytes(length);
  tvm::support::b64decode(state, bytes.data());

  const std::string name = tmpnam(NULL);
  auto file_name = name + ".so";
  std::unique_ptr<FILE, Deleter> pFile(fopen(file_name.c_str(), "wb"), Deleter(file_name));
  fwrite(bytes.data(), sizeof(u_char), length, pFile.get());
  fflush(pFile.get());

  std::string load_f_name = "runtime.module.loadfile_so";
  const PackedFunc* f = runtime::Registry::Get(load_f_name);
  ICHECK(f != nullptr) << "Loader for `.so` files is not registered,"
                       << " resolved to (" << load_f_name << ") in the global registry."
                       << "Ensure that you have loaded the correct runtime code, and"
                       << "that you are on the correct hardware architecture.";

  tvm::runtime::Module ret = (*f)(file_name, "");

  return ret;
}

/**
 * @brief A Torch's module which wraps TVM's OperatorModule Class.
 * The basic forward function calling TVM's runtime is provided.
 * The TVM module can be serialized/deserialized as a Torch module.
 */
class OperatorModuleWrapper : public torch::jit::CustomClassHolder {
 public:
  OperatorModuleWrapper() { runtime_module = ThreadLocalStore::ThreadLocal()->mod; }

  void forward(const c10::List<at::Tensor>& inputs) {
    int input_length = inputs.size();

    std::vector<DLManagedTensor*> tensors;

    for (int i = 0; i < input_length; ++i) tensors.push_back(toDLPack(inputs[i]));

    tvm::runtime::PackedFunc run = runtime_module.GetFunction("__tvm_main__");

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

  SerializationType Serialize() { return serialize(runtime_module); }

  explicit OperatorModuleWrapper(SerializationType state) { runtime_module = deserialize(state); }

 private:
  tvm::runtime::Module runtime_module;
};

tvm::Device getDevice(const at::Tensor& tensor) {
  tvm::Device dev;
  dev.device_id = tensor.get_device();
  switch (tensor.device().type()) {
    case at::DeviceType::CPU:
      dev.device_type = DLDeviceType::kDLCPU;
      if (dev.device_id == -1) {
        /*
         * In PyTorch the device ID for cpu is -1, sometimes causing error during tuning
         * Thus we manually set the device ID as 0 for avoiding potentially error of index out of
         * bounds
         */
        dev.device_id = 0;
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

/**
 * @brief A Torch's module which wraps TVM's GraphExecutorFactory Class.
 * The basic forward function calling TVM's runtime is provided.
 * The TVM module can be serialized/deserialized as a Torch module.
 */
class GraphExecutorFactoryWrapper : public torch::jit::CustomClassHolder {
 public:
  explicit GraphExecutorFactoryWrapper(tvm::runtime::Module executor_factory)
      : executor_factory_(executor_factory) {
    CHECK(executor_factory_->IsInstance<runtime::GraphExecutorFactory>())
        << "module is not an instance of GraphExecutorFactory";
  }

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

  SerializationType Serialize() { return serialize(executor_factory_); }

  explicit GraphExecutorFactoryWrapper(SerializationType state) {
    executor_factory_ = deserialize(state);
  }

 private:
  tvm::runtime::Module executor_factory_;
  tvm::runtime::Module executor_;
};

TVM_REGISTER_GLOBAL("tvmtorch.save_runtime_mod").set_body_typed([](tvm::runtime::Module mod) {
  ThreadLocalStore::ThreadLocal()->mod = mod;
});

TORCH_LIBRARY(tvm_torch, m) {
  m.class_<OperatorModuleWrapper>("OperatorModuleWrapper")
      .def(torch::init<>())
      .def("forward", &OperatorModuleWrapper::forward)
      .def_pickle(
          [](const c10::intrusive_ptr<OperatorModuleWrapper>& self) -> SerializationType {
            return self->Serialize();
          },
          [](SerializationType state) {
            return c10::make_intrusive<OperatorModuleWrapper>(state);
          });
  m.class_<GraphExecutorFactoryWrapper>("GraphExecutorFactoryWrapper")
      .def(torch::init<>())
      .def("forward", &GraphExecutorFactoryWrapper::forward)
      .def_pickle(
          [](const c10::intrusive_ptr<GraphExecutorFactoryWrapper>& self) -> SerializationType {
            return self->Serialize();
          },
          [](SerializationType state) {
            return c10::make_intrusive<GraphExecutorFactoryWrapper>(state);
          });
}

}  // namespace contrib
}  // namespace tvm
