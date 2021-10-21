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
#include <qurt_hvx.h>
}

#include <algorithm>
#include <memory>
#include <string>

#include "launcher_core.h"
#include "launcher_rpc.h"

static std::unique_ptr<Model> TheModel;

static AEEResult error_too_small(const std::string& func_name, const std::string& value_name,
                                 int given, int needed) {
  FARF(ERROR, "%s: %s value too small (%d), need at least %d", func_name.c_str(),
       value_name.c_str(), given, needed);
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
    FARF(ERROR, "%s: model already loaded, unload first", __func__);
    return AEE_EUNABLETOLOAD;
  }

  tvm::runtime::Module module = load_module(module_path);
  tvm::runtime::Module executor = create_graph_executor(graph_json, module, Model::device());

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
      get_module_func(TheModel->graph_executor, "get_num_inputs");
  *num_inputs = get_num_inputs();
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(launcher_rpc_set_input)(remote_handle64 handle, int input_idx,
                                                const unsigned char* input_meta, int meta_size,
                                                const unsigned char* input_value, int value_size) {
  if (!TheModel) {
    // No model created.
    FARF(ERROR, "%s: no model created", __func__);
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

  tvm::runtime::PackedFunc set_input = get_module_func(TheModel->graph_executor, "set_input");
  set_input(input_idx, input);

  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(launcher_rpc_get_num_outputs)(remote_handle64 handle, int* num_outputs) {
  if (!TheModel) {
    // No model created.
    return AEE_EBADSTATE;
  }

  tvm::runtime::PackedFunc get_num_outputs =
      get_module_func(TheModel->graph_executor, "get_num_outputs");
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

  tvm::runtime::PackedFunc get_output = get_module_func(TheModel->graph_executor, "get_output");
  tvm::runtime::NDArray output = get_output(output_idx);

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

    auto data = reinterpret_cast<decltype(output_value)>(output->data);
    std::copy(data, data + data_size, output_value);
  }

  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(launcher_rpc_run)(remote_handle64 handle, uint64_t* pcycles,
                                          uint64_t* usecs) {
  if (!TheModel) {
    // No model created.
    FARF(ERROR, "%s: no model created", __func__);
    return AEE_EBADSTATE;
  }

  // Reserve HVX.
  int res = qurt_hvx_reserve(QURT_HVX_RESERVE_ALL_AVAILABLE);
  switch (res) {
    case QURT_HVX_RESERVE_NOT_SUPPORTED:
    case QURT_HVX_RESERVE_NOT_SUCCESSFUL:
      FARF(ERROR, "error reserving HVX: %u", res);
      return AEE_EFAILED;
    default:
      break;
  }
  // Lock HVX.
  int lck = qurt_hvx_lock(QURT_HVX_MODE_128B);
  if (lck != 0) {
    FARF(ERROR, "error locking HVX: %u", lck);
    return AEE_EFAILED;
  }

  uint64_t us_begin = HAP_perf_get_time_us();
  uint64_t pc_begin = HAP_perf_get_pcycles();

  TheModel->run();

  uint64_t pc_end = HAP_perf_get_pcycles();
  uint64_t us_end = HAP_perf_get_time_us();
  *pcycles = pc_end - pc_begin;
  *usecs = us_end - us_begin;

  // Unlock HVX.
  int unl = qurt_hvx_unlock();
  if (unl != 0) {
    FARF(ERROR, "error unlocking HVX: %u", unl);
    return AEE_EFAILED;
  }
  // Release HVX.
  int rel = qurt_hvx_cancel_reserve();
  if (rel != 0) {
    FARF(ERROR, "error canceling HVX reservation: %u", rel);
    return AEE_EFAILED;
  }

  return AEE_SUCCESS;
}
