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
#include "../../../support/base64.h"
#include "runtime_bridge.h"

namespace tvm {
namespace contrib {

/*
 * TVM's FFI for passing module from python to C++
 */
struct ThreadLocalStore {
  tvm::runtime::Module mod;
  static ThreadLocalStore* ThreadLocal() {
    thread_local ThreadLocalStore tls;
    return &tls;
  }
};

TVM_REGISTER_GLOBAL("tvmtorch.save_runtime_mod").set_body_typed([](tvm::runtime::Module mod) {
  ThreadLocalStore::ThreadLocal()->mod = mod;
});

/*
 * Convert NDArray to DLPack extend tensor. It should be zero-cost.
 * @param src Pointer to NDArray
 * @return DLPack extended tensor
 */
DLPackTensorExt CreateDLpackTensorExt(tvm::runtime::NDArray* src) {
  auto is_bool = src->DataType().is_bool();
  DLManagedTensor* tensor;
  if (is_bool) {
    // If we change DLDataType{kDLInt, 8, 1} to DataType::Bool()
    // we will get `RuntimeError: Unsupported kUInt bits 1`
    auto tmp = src->CreateView(src->Shape(), DLDataType{kDLInt, 8, 1});
    tensor = tmp.ToDLPack();
  } else {
    tensor = src->ToDLPack();
  }
  DLPackTensorExt ret{tensor, is_bool};
  return ret;
}

/*
 * Create an NDArray with boolean type. (One memory copy)
 * @param src DLpack extended tensor
 * @return a new NDArray
 */
tvm::runtime::NDArray CreateBoolNDarray(DLPackTensorExt* src) {
  auto& tensor = src->dl_managed_tensor->dl_tensor;
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < tensor.ndim; i++) {
    shape.push_back(tensor.shape[i]);
  }
  auto ret = tvm::runtime::NDArray::Empty(shape, DataType::Bool(), tensor.device);
  ret.CopyFrom(&src->dl_managed_tensor->dl_tensor);
  return std::move(ret);
}

bool IsZeroCopy(DLPackTensorExt* src) {
  auto& dl_tensor = src->dl_managed_tensor->dl_tensor;
  return tvm::runtime::NDArray::AbilityOfZeroCopyForDLTensor(&dl_tensor, dl_tensor.device);
}

/*
 * Create an NDArray from DLpack extended tensor.
 * @param src DLpack extended tensor
 * @return a new NDArray
 */
tvm::runtime::NDArray NDarrayFromDLpack(DLPackTensorExt* src) {
  using tvm::runtime::NDArray;

  NDArray array;
  auto& dl_tensor = src->dl_managed_tensor->dl_tensor;
  if (src->is_bool) {
    // one memory copy
    // the code is similar to NewFromDLTensor except for the type
    array = CreateBoolNDarray(src);
  } else if (IsZeroCopy(src)) {
    array = NDArray::FromExternalDLTensor(src->dl_managed_tensor->dl_tensor);
  } else {
    // one memory copy
    array = NDArray::NewFromDLTensor(&dl_tensor, dl_tensor.device);
  }
  return array;
}

}  // namespace contrib
}  // namespace tvm

extern "C" {

struct TVMContribTorchRuntimeModule {
  tvm::runtime::Module mod;

  explicit TVMContribTorchRuntimeModule(tvm::runtime::Module& mod) : mod(mod) {}
};

bool tvm_contrib_torch_tensor_ability_of_zero_copy(DLPackTensorExt* src) {
  return (!src->is_bool) && (tvm::contrib::IsZeroCopy(src));
}

TVMContribTorchRuntimeModule* tvm_contrib_torch_get_last_saved_runtime_module() {
  return new TVMContribTorchRuntimeModule(tvm::contrib::ThreadLocalStore::ThreadLocal()->mod);
}

void tvm_contrib_torch_operator_module_forward(TVMContribTorchRuntimeModule* runtime_module,
                                               DLPackTensorExt* inputs, size_t input_size) {
  tvm::runtime::PackedFunc run = runtime_module->mod.GetFunction("__tvm_main__");

  std::vector<TVMValue> tvm_values(input_size);
  std::vector<int> tvm_type_codes(input_size);
  tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());

  std::vector<tvm::runtime::NDArray> input_cache(input_size);

  for (size_t k = 0; k < input_size; ++k) {
    auto datum = tvm::contrib::NDarrayFromDLpack(&inputs[k]);  // could have one memory copy
    input_cache[k] = datum;  // we keep the datum in a vector for future use, otherwise the datum
                             // will be freed after the loop
    setter(k, datum);
  }

  run.CallPacked(tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), input_size),
                 nullptr);

  for (size_t k = 0; k < input_size; ++k) {
    if (!tvm_contrib_torch_tensor_ability_of_zero_copy(&inputs[k]))
      input_cache[k].CopyTo(&inputs[k].dl_managed_tensor->dl_tensor);
  }
}

TVMContribTorchRuntimeModule* tvm_contrib_torch_create_graph_runtime_module(
    TVMContribTorchRuntimeModule* graph_executor_factory, DLManagedTensor* input_example) {
  tvm::runtime::PackedFunc built_module = graph_executor_factory->mod.GetFunction("default");
  tvm::Device device_info = input_example->dl_tensor.device;
  tvm::runtime::Module runtime_module = built_module(device_info);
  return new TVMContribTorchRuntimeModule(runtime_module);
}

size_t tvm_contrib_torch_graph_executor_module_forward(TVMContribTorchRuntimeModule* runtime_module,
                                                       DLPackTensorExt* inputs, size_t input_size,
                                                       DLPackTensorExt** outputs) {
  tvm::runtime::PackedFunc run = runtime_module->mod.GetFunction("run");
  tvm::runtime::PackedFunc set_input = runtime_module->mod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = runtime_module->mod.GetFunction("get_output");
  tvm::runtime::PackedFunc get_num_outputs = runtime_module->mod.GetFunction("get_num_outputs");

  for (size_t k = 0; k < input_size; ++k) {
    set_input(k, &inputs[k].dl_managed_tensor->dl_tensor);
  }

  run();

  int64_t output_length = get_num_outputs();

  DLPackTensorExt* outputs_ptr = new DLPackTensorExt[output_length];
  *outputs = outputs_ptr;

  for (int64_t k = 0; k < output_length; ++k) {
    tvm::runtime::NDArray results = get_output(k);
    outputs_ptr[k] = tvm::contrib::CreateDLpackTensorExt(&results);
  }

  return output_length;
}

inline size_t b64strlen(const std::string b64str) {
  ICHECK(b64str.size() % 4 == 0) << "invalid base64 encoding";
  size_t length = b64str.size() / 4 * 3;
  if (b64str[b64str.size() - 2] == '=') {
    length -= 2;
  } else if (b64str[b64str.size() - 1] == '=') {
    length -= 1;
  }
  return length;
}

inline void b64decode(const std::string b64str, uint8_t* ret) {
  size_t index = 0;
  const auto length = b64str.size();
  for (size_t i = 0; i < length; i += 4) {
    int8_t ch0 = tvm::support::base64::DecodeTable[(int32_t)b64str[i]];
    int8_t ch1 = tvm::support::base64::DecodeTable[(int32_t)b64str[i + 1]];
    int8_t ch2 = tvm::support::base64::DecodeTable[(int32_t)b64str[i + 2]];
    int8_t ch3 = tvm::support::base64::DecodeTable[(int32_t)b64str[i + 3]];
    uint8_t st1 = (ch0 << 2) + (ch1 >> 4);
    ret[index++] = st1;
    if (b64str[i + 2] != '=') {
      uint8_t st2 = ((ch1 & 0b1111) << 4) + (ch2 >> 2);
      ret[index++] = st2;
      if (b64str[i + 3] != '=') {
        uint8_t st3 = ((ch2 & 0b11) << 6) + ch3;
        ret[index++] = st3;
      }
    }
  }
  ICHECK(b64strlen(b64str) == index) << "base64 decoding fails";
}

/*!
 * \brief Export TVM runtime module to base64 stream including its submodules.
 * Note that this targets modules that are binary serializable and DSOExportable.
 * \param module The runtime module to export
 * \return std::string The content of exported file
 */
std::string ExportModuleToBase64(tvm::runtime::Module module) {
  static const tvm::runtime::PackedFunc* f_to_str =
      tvm::runtime::Registry::Get("export_runtime_module");
  ICHECK(f_to_str) << "IndexError: Cannot find the packed function "
                      "`export_runtime_module` in the global registry";
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

/*!
 * \brief Import TVM runtime module from base64 stream
 * Note that this targets modules that are binary serializable and DSOExportable.
 * \param base64str base64 stream, which are generated by `ExportModuleToBase64`.
 * \return runtime::Module runtime module constructed from the given stream
 */
tvm::runtime::Module ImportModuleFromBase64(std::string base64str) {
  auto length = b64strlen(base64str);

  std::vector<uint8_t> bytes(length);  // bytes stream
  b64decode(base64str, bytes.data());

  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream datetime;
  datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%X");
  const std::string file_name = "tmp-module-" + datetime.str() + ".so";
  LOG(INFO) << file_name;
  std::unique_ptr<FILE, Deleter> pFile(fopen(file_name.c_str(), "wb"), Deleter(file_name));
  fwrite(bytes.data(), sizeof(uint8_t), length, pFile.get());
  fflush(pFile.get());

  std::string load_f_name = "runtime.module.loadfile_so";
  const tvm::runtime::PackedFunc* f = tvm::runtime::Registry::Get(load_f_name);
  ICHECK(f != nullptr) << "Loader for `.so` files is not registered,"
                       << " resolved to (" << load_f_name << ") in the global registry."
                       << "Ensure that you have loaded the correct runtime code, and"
                       << "that you are on the correct hardware architecture.";
  tvm::runtime::Module ret = (*f)(file_name, "");
  return ret;
}

char* tvm_contrib_torch_encode(TVMContribTorchRuntimeModule* runtime_module) {
  std::string std = ExportModuleToBase64(runtime_module->mod);
  char* ret = new char[std.length() + 1];
  snprintf(ret, std.length() + 1, "%s", std.c_str());
  return ret;
}

TVMContribTorchRuntimeModule* tvm_contrib_torch_decode(const char* state) {
  tvm::runtime::Module ret = ImportModuleFromBase64(state);
  return new TVMContribTorchRuntimeModule(ret);
}

void tvm_contrib_torch_free_runtime_module(TVMContribTorchRuntimeModule* module_ptr) {
  delete module_ptr;
}

void tvm_contrib_torch_free_dlpack_tensor_ext_array(DLPackTensorExt* dlpack_ptr) {
  delete[] dlpack_ptr;
}

void tvm_contrib_torch_free_encoding(char* encoding) { delete[] encoding; }
}
