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

/*!
 * \brief Main entry point for
 * \param model Model descriptor structure to reference for runtime information
 * \param inputs Pointer to input pointer(s)
 * \param outputs Pointer to output pointer(s)
 * \param context Context information to be passed through to operators
 * \return tvm_status_t containing success or errors from the model run
 */
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/internal/aot_executor/aot_executor.h>

tvm_crt_error_t tvm_runtime_run(const tvm_model_t* model, void** inputs, void** outputs) {
  static DLDevice fake_device = {kDLCPU, 0};
  static int64_t fake_dims = 0;
  static int64_t fake_shape = {0};

  DLTensor tensors[model->num_input_tensors + model->num_output_tensors];     // NOLINT
  TVMValue tvm_values[model->num_input_tensors + model->num_output_tensors];  // NOLINT
  int32_t tvm_typeids[model->num_input_tensors + model->num_output_tensors];  // NOLINT

  for (int i = 0; i < model->num_input_tensors; i++) {
    tensors[i] = (DLTensor){
        .device = fake_device,
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
        .device = fake_device,
        .data = outputs[i],
        .shape = &fake_shape,
        .ndim = fake_dims,
        .byte_offset = 0,
        .strides = NULL,
    };
    tvm_values[model->num_input_tensors + i].v_handle = &tensors[model->num_input_tensors + i];
  }

  return model->run_func(tvm_values, tvm_typeids, 0, NULL, 0, NULL);
}
