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

// LINT_C_FILE

/*!
 * \file src/runtime/crt/aot/tvm_executor.c
 * \brief Internal implementation of the AOT Executor
 */

#include "tvm_executor.h"

#include <dlpack/dlpack.h>

#include "tvm_backend.h"
#include "tvm_error.h"

tvm_workspace_t* tvm_runtime_workspace;

tvm_crt_error_t tvm_runtime_run(const tvm_model_t* model, void** inputs, void** outputs,
                                tvm_context_t* context) {
  static DLContext fake_ctx = {kDLCPU, 0};
  static int64_t fake_dims = 0;
  static int64_t fake_shape = {0};

  DLTensor tensors[model->num_input_tensors + model->num_output_tensors];     // NOLINT
  TVMValue tvm_values[model->num_input_tensors + model->num_output_tensors];  // NOLINT
  int32_t tvm_typeids[model->num_input_tensors + model->num_output_tensors];  // NOLINT

  for (int i = 0; i < model->num_input_tensors; i++) {
    tensors[i] = (DLTensor){
        .ctx = fake_ctx,
        .data = inputs[i],
        .shape = &fake_shape,
        .ndim = fake_dims,
        .byte_offset = 0,
        .strides = NULL,
    };
    tvm_values[i].v_handle = &tensors[i];
  }

  for (int i = 0; i < model->num_output_tensors; i++) {
    tensors[model->num_input_tensors + i] = (DLTensor){
        .ctx = fake_ctx,
        .data = outputs[i],
        .shape = &fake_shape,
        .ndim = fake_dims,
        .byte_offset = 0,
        .strides = NULL,
    };
    tvm_values[model->num_input_tensors + i].v_handle = &tensors[model->num_input_tensors + i];
  }

  return model->run_func(&tvm_values, &tvm_typeids, 0, NULL, 0, context);
}

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,
                               int dtype_bits_hint) {
  uint32_t offset = (~nbytes + 1) & (TVM_RUNTIME_ALLOC_ALIGNMENT - 1);
  uint8_t* current_alloc = tvm_runtime_workspace->next_alloc;
  uint8_t* next_alloc = tvm_runtime_workspace->next_alloc + nbytes + offset;
  uint8_t* workspace_end = tvm_runtime_workspace->workspace + tvm_runtime_workspace->workspace_size;

  if (next_alloc > workspace_end) {
    return NULL;
  }

  tvm_runtime_workspace->next_alloc = next_alloc;
  return current_alloc;
}

tvm_crt_error_t TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  tvm_runtime_workspace->next_alloc = ptr;
  return 0;
}
