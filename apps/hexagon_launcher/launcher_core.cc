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

#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

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
  out << "ndim=" << ndim << ", dtype=" << tvm::runtime::DLDataType2String(dtype) << ", shape=";
  for (int i = 0; i != ndim; ++i) {
    out << shape[i];
    if (i + 1 < ndim) {
      out << 'x';
    }
  }
  return out.str();
}

void TensorConfig::Load(dmlc::JSONReader* reader) {
  reader->BeginObject();
  std::string key;
  while (!bad && reader->NextObjectItem(&key)) {
    if (key == file_key) {
      reader->Read(&file_name);
    } else if (key == shape_key) {
      reader->Read(&shape);
      if (shape.empty()) {
        std::cout << "error: empty shape\n";
        bad = true;
      }
    } else if (key == dtype_key) {
      reader->Read(&dtype);
    } else {
      std::cout << "unknown tensor config key: " << key << '\n';
      bad = true;
    }
  }
}

void TensorConfig::Save(dmlc::JSONWriter* writer) const {
  writer->BeginObject(true);
  writer->WriteObjectKeyValue(file_key, file_name);
  writer->WriteObjectKeyValue(shape_key, shape);
  writer->WriteObjectKeyValue(dtype_key, dtype);
  writer->EndObject();
}

void ModelConfig::Load(dmlc::JSONReader* reader) {
  reader->BeginObject();
  std::string key;
  while (!bad && reader->NextObjectItem(&key)) {
    if (key == "model-library") {
      reader->Read(&model_library);
    } else if (key == "model-json") {
      reader->Read(&model_json);
    } else if (key == "inputs") {
      reader->Read(&inputs);
      bad = std::any_of(inputs.begin(), inputs.end(), [](auto t) { return t.bad; });
    } else {
      std::cout << "unknown model config key: " << key << '\n';
      bad = true;
    }
  }
}

void OutputConfig::Save(dmlc::JSONWriter* writer) const {
  writer->BeginObject(true);
  writer->WriteObjectKeyValue("pcycles", pcycles);
  writer->WriteObjectKeyValue("usecs", usecs);
  writer->WriteObjectKeyValue("outputs", outputs);
  writer->EndObject();
}

bool read_model_config(const std::string& file_name, ModelConfig* model_config) {
  if (model_config == nullptr) {
    return false;
  }
  std::ifstream mfc(file_name);
  if (!mfc.is_open()) {
    return false;
  }
  dmlc::JSONReader reader(&mfc);
  model_config->Load(&reader);
  if (model_config->bad || !mfc) {
    return false;
  }
  return true;
}

bool write_output_config(const std::string& file_name, OutputConfig* output_config) {
  std::ofstream ofc(file_name);
  if (!ofc.is_open()) {
    return false;
  }
  dmlc::JSONWriter writer(&ofc);
  output_config->Save(&writer);
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

const tvm::runtime::PackedFunc get_runtime_func(const std::string& name) {
  if (const tvm::runtime::PackedFunc* pf = tvm::runtime::Registry::Get(name)) {
    return *pf;
  }
  return tvm::runtime::PackedFunc();
}

const tvm::runtime::PackedFunc get_module_func(tvm::runtime::Module module,
                                               const std::string& name) {
  return module.GetFunction(name, false);
}

void reset_device_api() {
  const tvm::runtime::PackedFunc api = get_runtime_func("device_api.hexagon");
  tvm::runtime::Registry::Register("device_api.cpu", true).set_body(api);
}

tvm::runtime::Module load_module(const std::string& file_name) {
  static const tvm::runtime::PackedFunc loader =
      get_runtime_func("runtime.module.loadfile_hexagon");
  tvm::runtime::TVMRetValue rv = loader(file_name);
  if (rv.type_code() == kTVMModuleHandle) {
    ICHECK_EQ(rv.type_code(), kTVMModuleHandle)
        << __func__ << ": loaded " << file_name << ", but did not get module handle";
    return rv.operator tvm::runtime::Module();
  }
  return tvm::runtime::Module();
}

std::ostream& operator<<(std::ostream& os, const tvm::Array<tvm::String>& strings) {
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

  const tvm::runtime::PackedFunc create_executor = get_runtime_func(launcher_name);
  uint64_t device_type = device.device_type;
  uint64_t device_id = device.device_id;

  if (graph_json.empty()) {
    LOG(ERROR) << __func__ << ": graph executor requires graph JSON";
    return tvm::runtime::Module();
  }
  tvm::runtime::TVMRetValue rv = create_executor(graph_json, graph_module, device_type, device_id);
  return rv.operator tvm::runtime::Module();
}

tvm::runtime::Module create_aot_executor(tvm::runtime::Module factory_module, tvm::Device device) {
  tvm::runtime::PackedFunc list_modules = get_module_func(factory_module, "list_module_names");
  tvm::Array<tvm::String> module_names = list_modules();
  if (module_names.size() != 1) {
    LOG(WARNING) << __func__ << ": expecting single module, got: " << module_names << ", using "
                 << module_names[0];
  }
  tvm::runtime::PackedFunc f = get_module_func(factory_module, module_names[0]);
  if (f.get() == nullptr) {
    LOG(ERROR) << __func__ << ": failed to obtain function " << module_names[0];
    return tvm::runtime::Module();
  }
  return f(device);
}
