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

extern "C" {
#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <qurt_error.h>
}

#include <tvm/runtime/object.h>

#include <algorithm>
#include <memory>
#include <string>

#include "launcher_core.h"
#include "launcher_rpc.h"

static std::unique_ptr<Model> TheModel;

static AEEResult error_too_small(const std::string& func_name, const std::string& value_name,
                                 int given, int needed) {
  LOG(ERROR) << func_name.c_str() << ": " << value_name.c_str() << " value too small (" << given
             << "), need at least " << needed;
  return AEE_EBADPARM;
}

int __QAIC_HEADER(launcher_rpc_open)(const char* uri, remote_handle64* handle) {
  *handle = 0;  // Just use any value.
  reset_device_api();
  return AEE_SUCCESS;
}

int __QAIC_HEADER(launcher_rpc_close)(remote_handle64 handle) {
  // Comment to stop clang-format from single-lining this function.
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(launcher_rpc_load)(remote_handle64 handle, const char* module_path,
                                           const char* graph_json) {
  if (TheModel) {
    // Need to unload first.
    LOG(ERROR) << __func__ << ": model already loaded, unload first";
    return AEE_EUNABLETOLOAD;
  }

  tvm::runtime::Module module = load_module(module_path);
  std::string module_type = module->type_key();
  tvm::runtime::Module executor;
  if (module_type == "AotExecutorFactory") {
    executor = create_aot_executor(module, Model::external());
  } else if (module_type == "library") {
    // We're not expecting "GraphExecutorFactory" here.
    executor = create_graph_executor(graph_json, module, Model::device());
  } else {
    LOG(ERROR) << __func__ << ": unexpected module type: " << module_type;
    // Fall through.
  }

  if (executor.get() == nullptr) {
    LOG(ERROR) << __func__ << ": failed to create executor for module" << module_path;
    return AEE_EUNABLETOLOAD;
  }

  TheModel = std::make_unique<Model>(executor, module, graph_json);
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(launcher_rpc_unload)(remote_handle64 handle) {
  if (TheModel) {
    TheModel.reset();
  }
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(launcher_rpc_get_num_inputs)(remote_handle64 handle, int* num_inputs) {
  if (!TheModel) {
    // No model created.
    return AEE_EBADSTATE;
  }

  tvm::runtime::PackedFunc get_num_inputs =
      get_module_func(TheModel->model_executor, "get_num_inputs");
  *num_inputs = get_num_inputs();
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(launcher_rpc_set_input)(remote_handle64 handle, int input_idx,
                                                const unsigned char* input_meta, int meta_size,
                                                const unsigned char* input_value, int value_size) {
  if (!TheModel) {
    // No model created.
    LOG(ERROR) << __func__ << ": no model created";
    return AEE_EBADSTATE;
  }

  const auto* meta = reinterpret_cast<const tensor_meta*>(input_meta);
  if (meta_size < meta->meta_size()) {
    return error_too_small(__func__, "meta_size", meta_size, meta->meta_size());
  }
  if (value_size < meta->data_size()) {
    return error_too_small(__func__, "value_size", value_size, meta->data_size());
  }

  DLTensor tensor{
      const_cast<unsigned char*>(input_value),
      Model::external(),
      meta->ndim,
      meta->dtype,
      const_cast<int64_t*>(meta->shape),
      /*strides*/ nullptr,
      /*byte_offset*/ 0,
  };
  DLManagedTensor managed{tensor, /*manager_ctx*/ nullptr, /*deleter*/ nullptr};

  auto input = tvm::runtime::NDArray::FromDLPack(&managed);

  tvm::runtime::PackedFunc set_input = get_module_func(TheModel->model_executor, "set_input");
  set_input(input_idx, input);

  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(launcher_rpc_get_num_outputs)(remote_handle64 handle, int* num_outputs) {
  if (!TheModel) {
    // No model created.
    return AEE_EBADSTATE;
  }

  tvm::runtime::PackedFunc get_num_outputs =
      get_module_func(TheModel->model_executor, "get_num_outputs");
  *num_outputs = get_num_outputs();
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(launcher_rpc_get_output)(remote_handle64 handle, int output_idx,
                                                 unsigned char* output_meta, int meta_size,
                                                 unsigned char* output_value, int value_size) {
  if (!TheModel) {
    // No model created.
    return AEE_EBADSTATE;
  }
  if (meta_size < 0 || value_size < 0) {
    return AEE_EBADPARM;
  }
  if ((output_meta == nullptr && meta_size != 0) || (output_value == nullptr && value_size != 0)) {
    // If the pointer is null, the size must be 0.
    return AEE_EBADPARM;
  }

  tvm::runtime::PackedFunc get_output = get_module_func(TheModel->model_executor, "get_output");
  tvm::runtime::NDArray output = get_output(output_idx);

  std::vector<int64_t> shape_vec{output->shape, output->shape + output->ndim};

  auto* container = new tvm::runtime::NDArray::Container(
      static_cast<void*>(output_value), shape_vec, output->dtype, Model::external());
  container->SetDeleter([](tvm::Object* container) {
    delete static_cast<tvm::runtime::NDArray::Container*>(container);
  });

  tvm::runtime::NDArray host_output(tvm::runtime::GetObjectPtr<tvm::runtime::Object>(container));

  if (meta_size != 0) {
    auto* meta = reinterpret_cast<tensor_meta*>(output_meta);
    if (meta_size < meta->meta_size(output->ndim)) {
      return error_too_small(__func__, "meta_size", meta_size, meta->meta_size(output->ndim));
    }

    meta->ndim = output->ndim;
    meta->dtype = output->dtype;
    std::copy(&output->shape[0], &output->shape[output->ndim], meta->shape);
  }

  if (value_size != 0) {
    size_t data_size = tvm::runtime::GetDataSize(*output.operator->());
    if (value_size < data_size) {
      return error_too_small(__func__, "value_size", value_size, data_size);
    }

    host_output.CopyFrom(output);
  }

  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(launcher_rpc_run)(remote_handle64 handle, uint64_t* pcycles,
                                          uint64_t* usecs) {
  if (!TheModel) {
    // No model created.
    LOG(ERROR) << __func__ << ": no model created";
    return AEE_EBADSTATE;
  }

  uint64_t us_begin = HAP_perf_get_time_us();
  uint64_t pc_begin = HAP_perf_get_pcycles();

  TheModel->run();

  uint64_t pc_end = HAP_perf_get_pcycles();
  uint64_t us_end = HAP_perf_get_time_us();
  *pcycles = pc_end - pc_begin;
  *usecs = us_end - us_begin;

  return AEE_SUCCESS;
}
