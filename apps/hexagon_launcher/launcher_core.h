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

#ifndef TVM_RUNTIME_HEXAGON_LAUNCHER_LAUNCHER_CORE_H_
#define TVM_RUNTIME_HEXAGON_LAUNCHER_LAUNCHER_CORE_H_

#include <dlpack/dlpack.h>
#include <dmlc/json.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <string>
#include <vector>

struct tensor_meta {
  int ndim;
  DLDataType dtype;
  int64_t shape[];

  int meta_size() const { return meta_size(ndim); }
  int data_size() const {
    int size = tvm::runtime::DataType(dtype).bytes();
    for (int d = 0; d != ndim; ++d) {
      size *= shape[d];
    }
    return size;
  }

  static int meta_size(int ndim) { return sizeof(tensor_meta) + ndim * sizeof(int64_t); }

  std::string to_string() const;
};

struct TensorConfig {
  static const std::string file_key;
  static const std::string shape_key;
  static const std::string dtype_key;

  std::string file_name;
  std::vector<int> shape;
  std::string dtype;
  bool bad = false;

  void Load(dmlc::JSONReader* reader);
  void Save(dmlc::JSONWriter* writer) const;
};

struct ModelConfig {
  std::string model_library;
  std::string model_json;
  std::vector<TensorConfig> inputs;
  bool bad = false;

  void Load(dmlc::JSONReader* reader);
};

struct OutputConfig {
  uint64_t pcycles;
  uint64_t usecs;
  std::vector<TensorConfig> outputs;

  void Save(dmlc::JSONWriter* writer) const;
};

struct Model {
  Model(tvm::runtime::Module executor, tvm::runtime::Module module, std::string json);

  tvm::runtime::Module model_executor;
  tvm::runtime::Module graph_module;
  std::string graph_json;

  static tvm::Device device() { return tvm::Device{static_cast<DLDeviceType>(kDLHexagon), 0}; }
  static tvm::Device external() { return tvm::Device{static_cast<DLDeviceType>(kDLCPU), 0}; }

  tvm::runtime::PackedFunc run;
};

struct ExecutionSession {
  explicit ExecutionSession(bool lwp_json = false) : gen_lwp_json(lwp_json) {}

  template <typename T>
  T* alloc(size_t bytes, size_t align = 1) {
    return reinterpret_cast<T*>(alloc_mem(bytes, align));
  }
  void free(void* ptr) { free_mem(ptr); }

  virtual void* alloc_mem(size_t bytes, size_t align) = 0;
  virtual void free_mem(void* ptr) = 0;

  virtual bool load_model(const std::string& model_path, const std::string& model_json) = 0;
  virtual bool unload_model() = 0;

  virtual bool set_input(int input_idx, const tensor_meta* input_meta, const void* input_data) = 0;
  virtual bool run(uint64_t* pcycles, uint64_t* usecs) = 0;
  virtual bool get_num_outputs(int* num_outputs) = 0;
  virtual bool get_output(int output_idx, tensor_meta* output_meta, int meta_size,
                          void* output_data, int data_size) = 0;
  bool gen_lwp_json = false;
};

bool read_model_config(const std::string& file_name, ModelConfig* model_config);
bool write_output_config(const std::string& file_name, OutputConfig* output_config);

void reset_device_api();

tvm::runtime::Module load_module(const std::string& file_name);

const tvm::runtime::PackedFunc get_runtime_func(const std::string& name);
const tvm::runtime::PackedFunc get_module_func(tvm::runtime::Module module,
                                               const std::string& name);

tvm::runtime::Module create_aot_executor(tvm::runtime::Module factory_module, tvm::Device device);
tvm::runtime::Module create_graph_executor(const std::string& graph_json,
                                           tvm::runtime::Module graph_module, tvm::Device device);

#endif  // TVM_RUNTIME_HEXAGON_LAUNCHER_LAUNCHER_CORE_H_
