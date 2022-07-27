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
#include <dlpack/dlpack.h>
#include <dmlc/memory_io.h>
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
#include "runtime_bridge.h"

struct ThreadLocalStore {
  tvm::runtime::Module mod;
  static ThreadLocalStore* ThreadLocal() {
    thread_local ThreadLocalStore tls;
    return &tls;
  }
};

namespace tvm {
namespace contrib {

std::string serialize(tvm::runtime::Module module) {
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
        << "remove temporary file (" << file_name << ") unsuccessfully";
  }
  std::string file_name;
};

tvm::runtime::Module deserialize(std::string state) {
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

TVM_REGISTER_GLOBAL("tvmtorch.save_runtime_mod").set_body_typed([](tvm::runtime::Module mod) {
  ThreadLocalStore::ThreadLocal()->mod = mod;
});

tvm::runtime::NDArray NDArrayFromDlPackExt(DLPackTensorExt dlpack_ext) {
  using tvm::runtime::NDArray;

  NDArray array;
  auto& dl_tensor = dlpack_ext.dl_managed_tensor->dl_tensor;
  bool is_zero_copy =
      tvm::runtime::NDArray::AbilityOfZeroCopyForDLTensor(&dl_tensor, dl_tensor.device);
  if (is_zero_copy) {
    // Zero-copy if data pointer is aligned
    array = NDArray::FromDLPack(dlpack_ext.dl_managed_tensor);
  } else {
    // Copy if data pointer isn't aligned to the kAllocAlignment of TVM
    array = NDArray::NewFromDLTensor(&dl_tensor, dl_tensor.device);
    dlpack_ext.dl_managed_tensor->deleter(dlpack_ext.dl_managed_tensor);
  }
  if (dlpack_ext.is_bool) {
    auto result = tvm::runtime::NDArray::Empty(array.Shape(), DataType::Bool(), array->device);
    result.CopyFrom(array);
    return result;
  }

  return array;
}

}  // namespace contrib
}  // namespace tvm

extern "C" {

struct TVMContribTorchRuntimeModule {
  tvm::runtime::Module mod;

  TVMContribTorchRuntimeModule(tvm::runtime::Module mod) : mod(mod) {}
};

TVMContribTorchRuntimeModule* tvm_contrib_torch_get_last_saved_runtime_module() {
  return new TVMContribTorchRuntimeModule(ThreadLocalStore::ThreadLocal()->mod);
}

void tvm_contrib_torch_operator_module_forward(TVMContribTorchRuntimeModule* runtime_module,
                                               TensorList inputs, size_t input_size) {
  tvm::runtime::PackedFunc run = runtime_module->mod.GetFunction("__tvm_main__");

  std::vector<TVMValue> tvm_values(input_size);
  std::vector<int> tvm_type_codes(input_size);
  tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
  for (int k = 0; k < input_size; ++k) {
    // auto datum = tvm::contrib::NDArrayFromDlPackExt(inputs[k]);
    setter(k, &inputs[k]->dl_tensor);
  }

  run.CallPacked(tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), input_size),
                 nullptr);
}

tvm::Device getDeviceInfo(DLManagedTensor* input_device) {
  return {.device_type = input_device->dl_tensor.device.device_type,
          .device_id = input_device->dl_tensor.device.device_id};
}

int64_t tvm_contrib_torch_graph_executor_module_forward(TVMContribTorchRuntimeModule* graph_module,
                                                        TensorList inputs, size_t input_size,
                                                        TensorList* outputs) {
  tvm::runtime::PackedFunc built_module = graph_module->mod.GetFunction("default");
  tvm::runtime::Module runtime_module = built_module(getDeviceInfo(inputs[0]));
  tvm::runtime::PackedFunc run = runtime_module.GetFunction("run");
  tvm::runtime::PackedFunc set_input = runtime_module.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = runtime_module.GetFunction("get_output");
  tvm::runtime::PackedFunc get_num_outputs = runtime_module.GetFunction("get_num_outputs");

  for (int k = 0; k < input_size; ++k) {
    set_input(k, &inputs[k]->dl_tensor);
  }

  run();

  int64_t output_length = get_num_outputs();

  auto out_ptr = new DLManagedTensor*[output_length];
  *outputs = out_ptr;

  for (int k = 0; k < output_length; ++k) {
    tvm::runtime::NDArray results = get_output(k);
    auto tensor = results.ToDLPack();
    out_ptr[k] = tensor;
  }

  return output_length;
}

char* tvm_contrib_torch_encode(TVMContribTorchRuntimeModule* runtime_module) {
  auto std = tvm::contrib::serialize(runtime_module->mod);
  auto* ret = new char[std.length() + 1];
  strcpy(ret, std.c_str());
  ret[std.length()] = '\0';
  return ret;
}

TVMContribTorchRuntimeModule* tvm_contrib_torch_decode(const char* state) {
  auto ret = tvm::contrib::deserialize(state);
  return new TVMContribTorchRuntimeModule(ret);
}
}