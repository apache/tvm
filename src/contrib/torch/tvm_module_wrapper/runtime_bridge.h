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
 * \file runtime_bridge.h
 * \brief Util functions for pytorch tvm interaction.
 */
#ifndef TVM_CONTRIB_TORCH_TVM_MODULE_WRAPPER_RUNTIME_BRIDGE_H_
#define TVM_CONTRIB_TORCH_TVM_MODULE_WRAPPER_RUNTIME_BRIDGE_H_

extern "C" {

struct DLPackTensorExt {
  DLManagedTensor* dl_managed_tensor;
  bool is_bool;
};

struct TVMContribTorchRuntimeModule;

TVMContribTorchRuntimeModule* tvm_contrib_torch_get_last_saved_runtime_module();

void tvm_contrib_torch_free_runtime_module(TVMContribTorchRuntimeModule* module_ptr);

TVMContribTorchRuntimeModule* tvm_contrib_torch_create_graph_runtime_module(
    TVMContribTorchRuntimeModule* graph_module, DLManagedTensor* input_example);

void tvm_contrib_torch_operator_module_forward(TVMContribTorchRuntimeModule* runtime_module,
                                               DLPackTensorExt* inputs, size_t input_size);

size_t tvm_contrib_torch_graph_executor_module_forward(TVMContribTorchRuntimeModule* graph_module,
                                                       DLPackTensorExt* inputs, size_t input_size,
                                                       DLPackTensorExt** outputs);

char* tvm_contrib_torch_encode(TVMContribTorchRuntimeModule* runtime_module);

TVMContribTorchRuntimeModule* tvm_contrib_torch_decode(const char* state);

void tvm_contrib_torch_free_dlpack_tensor_ext_array(DLPackTensorExt*);

void tvm_contrib_torch_free_encoding(char* encoding);
}

#endif  // TVM_CONTRIB_TORCH_TVM_MODULE_WRAPPER_RUNTIME_BRIDGE_H_
