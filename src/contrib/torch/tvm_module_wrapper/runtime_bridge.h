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
#ifndef TVM_CONTRIB_TORCH_RUNTIME_BRIDGE_H_
#define TVM_CONTRIB_TORCH_RUNTIME_BRIDGE_H_

extern "C" {

typedef DLManagedTensor** TensorList;

struct DLPackTensorExt {
  DLManagedTensor* dl_managed_tensor;
  bool is_bool;
};


struct RuntimeModulePointer;

RuntimeModulePointer* get_last_saved_runtime_module();

void operator_module_forward(RuntimeModulePointer* runtime_module, TensorList inputs,
                             size_t input_size);

int64_t graph_executor_module_forward(RuntimeModulePointer* graph_module, TensorList inputs,
                                      size_t input_size,
                                      TensorList* outputs);

char* encode(RuntimeModulePointer* runtime_module);

RuntimeModulePointer* decode(const char* state);
}

#endif  // TVM_CONTRIB_TORCH_RUNTIME_BRIDGE_H_
