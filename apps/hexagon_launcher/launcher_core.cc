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

#include "launcher_core.h"

#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/c_backend_api.h>

#include <algorithm>
#include <fstream>
#include <ios>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

const std::string TensorConfig::file_key = "file";    // NOLINT(runtime/string)
const std::string TensorConfig::shape_key = "shape";  // NOLINT(runtime/string)
const std::string TensorConfig::dtype_key = "dtype";  // NOLINT(runtime/string)

std::string tensor_meta::to_string() const {
  std::stringstream out;
  out << "ndim=" << ndim << ", dtype=" << tvm::runtime::DLDataTypeToString(dtype) << ", shape=";
  for (int i = 0; i != ndim; ++i) {
    out << shape[i];
    if (i + 1 < ndim) {
      out << 'x';
    }
  }
  return out.str();
}

void TensorConfig::Load(tvm::ffi::json::Object obj) {
  namespace json = ::tvm::ffi::json;
  for (const auto& kv : obj) {
    std::string key = std::string(kv.first.cast<tvm::ffi::String>());
    if (key == file_key) {
      file_name = std::string(kv.second.cast<tvm::ffi::String>());
    } else if (key == shape_key) {
      auto arr = kv.second.cast<json::Array>();
      shape.clear();
      shape.reserve(arr.size());
      for (const auto& elem : arr) {
        shape.push_back(static_cast<int>(elem.cast<int64_t>()));
      }
      if (shape.empty()) {
        std::cout << "error: empty shape\n";
        bad = true;
      }
    } else if (key == dtype_key) {
      dtype = std::string(kv.second.cast<tvm::ffi::String>());
    } else {
      std::cout << "unknown tensor config key: " << key << '\n';
      bad = true;
    }
  }
}

tvm::ffi::json::Value TensorConfig::SaveToJSON() const {
  namespace json = ::tvm::ffi::json;
  json::Object obj;
  obj.Set(tvm::ffi::String(file_key), tvm::ffi::String(file_name));
  json::Array shape_arr;
  for (int v : shape) {
    shape_arr.push_back(static_cast<int64_t>(v));
  }
  obj.Set(tvm::ffi::String(shape_key), std::move(shape_arr));
  obj.Set(tvm::ffi::String(dtype_key), tvm::ffi::String(dtype));
  return obj;
}

void ModelConfig::Load(tvm::ffi::json::Object obj) {
  namespace json = ::tvm::ffi::json;
  for (const auto& kv : obj) {
    std::string key = std::string(kv.first.cast<tvm::ffi::String>());
    if (key == "model-library") {
      model_library = std::string(kv.second.cast<tvm::ffi::String>());
    } else if (key == "model-json") {
      model_json = std::string(kv.second.cast<tvm::ffi::String>());
    } else if (key == "inputs") {
      auto arr = kv.second.cast<json::Array>();
      inputs.clear();
      inputs.reserve(arr.size());
      for (const auto& elem : arr) {
        TensorConfig tc;
        tc.Load(elem.cast<json::Object>());
        inputs.push_back(std::move(tc));
      }
      bad = std::any_of(inputs.begin(), inputs.end(), [](auto t) { return t.bad; });
    } else {
      std::cout << "unknown model config key: " << key << '\n';
      bad = true;
    }
  }
}

tvm::ffi::json::Value OutputConfig::SaveToJSON() const {
  namespace json = ::tvm::ffi::json;
  json::Object obj;
  obj.Set(tvm::ffi::String("pcycles"), static_cast<int64_t>(pcycles));
  obj.Set(tvm::ffi::String("usecs"), static_cast<int64_t>(usecs));
  json::Array outputs_arr;
  for (const auto& tc : outputs) {
    outputs_arr.push_back(tc.SaveToJSON());
  }
  obj.Set(tvm::ffi::String("outputs"), std::move(outputs_arr));
  return obj;
}

bool read_model_config(const std::string& file_name, ModelConfig* model_config) {
  namespace json = ::tvm::ffi::json;
  if (model_config == nullptr) {
    return false;
  }
  std::ifstream mfc(file_name);
  if (!mfc.is_open()) {
    return false;
  }
  std::string content((std::istreambuf_iterator<char>(mfc)), std::istreambuf_iterator<char>());
  auto parsed = json::Parse(content);
  model_config->Load(parsed.cast<json::Object>());
  if (model_config->bad) {
    return false;
  }
  return true;
}

bool write_output_config(const std::string& file_name, OutputConfig* output_config) {
  namespace json = ::tvm::ffi::json;
  std::ofstream ofc(file_name);
  if (!ofc.is_open()) {
    return false;
  }
  ofc << std::string(json::Stringify(output_config->SaveToJSON(), 2));
  if (!ofc) {
    return false;
  }
  return true;
}

Model::Model(tvm::runtime::Module executor, tvm::runtime::Module module, std::string json)
    : model_executor(executor), graph_module(module), graph_json(json) {
  // Lookup "run" ahead of time to reduce overhead in the model execution.
  run = get_module_func(model_executor, "run");
}

const tvm::ffi::Function get_runtime_func(const std::string& name) {
  if (auto pf = tvm::ffi::Function::GetGlobal(name)) {
    return *pf;
  }
  return tvm::ffi::Function();
}

const tvm::ffi::Function get_module_func(tvm::runtime::Module module, const std::string& name) {
  return module->GetFunction(name, false).value_or(tvm::ffi::Function());
}

void reset_device_api() {
  const tvm::ffi::Function api = get_runtime_func("device_api.hexagon");
  tvm::ffi::Function::SetGlobal("device_api.cpu", api, true);
}

tvm::runtime::Module load_module(const std::string& file_name) {
  static const tvm::ffi::Function loader = get_runtime_func("ffi.Module.load_from_file.hexagon");
  tvm::ffi::Any rv = loader(file_name);
  if (rv.type_code() == kTVMModuleHandle) {
    TVM_FFI_ICHECK_EQ(rv.type_code(), kTVMModuleHandle)
        << __func__ << ": loaded " << file_name << ", but did not get module handle";
    return rv.operator tvm::runtime::Module();
  }
  return tvm::runtime::Module();
}

std::ostream& operator<<(std::ostream& os, const tvm::ffi::Array<tvm::ffi::String>& strings) {
  os << '[';
  for (int i = 0, e = strings.size(); i != e; ++i) {
    if (i != 0) os << ',';
    os << static_cast<std::string>(strings[i]);
  }
  os << ']';
  return os;
}

tvm::runtime::Module create_graph_executor(const std::string& graph_json,
                                           tvm::runtime::Module graph_module, tvm::Device device) {
  std::string launcher_name = "tvm.graph_executor.create";

  const tvm::ffi::Function create_executor = get_runtime_func(launcher_name);
  uint64_t device_type = device.device_type;
  uint64_t device_id = device.device_id;

  if (graph_json.empty()) {
    LOG(ERROR) << __func__ << ": graph executor requires graph JSON";
    return tvm::runtime::Module();
  }
  tvm::ffi::Any rv = create_executor(graph_json, graph_module, device_type, device_id);
  return rv.operator tvm::runtime::Module();
}

tvm::runtime::Module create_aot_executor(tvm::runtime::Module factory_module, tvm::Device device) {
  tvm::ffi::Function list_modules = get_module_func(factory_module, "list_module_names");
  tvm::ffi::Array<tvm::ffi::String> module_names = list_modules();
  if (module_names.size() != 1) {
    LOG(WARNING) << __func__ << ": expecting single module, got: " << module_names << ", using "
                 << module_names[0];
  }
  tvm::ffi::Function f = get_module_func(factory_module, module_names[0]);
  if (f.get() == nullptr) {
    LOG(ERROR) << __func__ << ": failed to obtain function " << module_names[0];
    return tvm::runtime::Module();
  }
  return f(device);
}
