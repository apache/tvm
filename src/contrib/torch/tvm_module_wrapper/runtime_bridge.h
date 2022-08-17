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

/*
 * DLPack data structure extend with `is_bool` flag.
 * DLPack haven't support boolean tensor
 * (https://github.com/pytorch/pytorch/blob/4618371da56c887195e2e1d16dad2b9686302800/aten/src/ATen/DLConvertor.cpp#L42),
 * thus a boolean tensor will be regarded as a UInt8 tensor
 * (https://github.com/apache/tvm/blob/de124862714e747764aa8b7f41a90bcb25f3c6a8/python/tvm/_ffi/runtime_ctypes.py#L91).
 */
struct DLPackTensorExt {
  DLManagedTensor* dl_managed_tensor;
  bool is_bool;
};

/*
 * A wrapper pointing to TVM runtime module.
 */
struct TVMContribTorchRuntimeModule;

/*
 * Obtain a saved runtime module passed by TVM FFI.
 * @return A TVM runtime module wrapper.
 */
TVMContribTorchRuntimeModule* tvm_contrib_torch_get_last_saved_runtime_module();

/*
 * Delete TVMContribTorchRuntimeModule pointer.
 */
void tvm_contrib_torch_free_runtime_module(TVMContribTorchRuntimeModule* module_ptr);

/*
 * Obtain ExecutorFactory runtime module from ExecutorFactory class.
 * @param graph_executor_factory ExecutorFactory class
 * @param input_example For obtaining device information
 * @return ExecutorFactory TVM runtime module wrapper
 */
TVMContribTorchRuntimeModule* tvm_contrib_torch_create_graph_runtime_module(
    TVMContribTorchRuntimeModule* graph_executor_factory, DLManagedTensor* input_example);

/*
 * Forward method for OperatorModuleWrapper.
 * @param runtime_module TVM runtime module wrapper
 * @param inputs Array pointer of the input tensors
 * @param input_size The number of input tensors
 */
void tvm_contrib_torch_operator_module_forward(TVMContribTorchRuntimeModule* runtime_module,
                                               DLPackTensorExt* inputs, size_t input_size);

/*
 * Forward method for GraphExecutorFactoryWrapper.
 * @param graph_executor_factory TVM runtime module wrapper
 * @param inputs Array pointer of the input tensors
 * @param input_size The number of input tensors
 * @param outputs The resulting output tensors pointer
 * @return The number of output tensors
 */
size_t tvm_contrib_torch_graph_executor_module_forward(
    TVMContribTorchRuntimeModule* graph_executor_factory, DLPackTensorExt* inputs,
    size_t input_size, DLPackTensorExt** outputs);

/*
 * Encode TVM runtime module.
 * @param runtime_module TVM runtime module wrapper
 * @return The encoding stream (char array)
 */
char* tvm_contrib_torch_encode(TVMContribTorchRuntimeModule* runtime_module);

/*
 * Decode TVM runtime module.
 * @param state The encoding stream (char array) of TVM runtime module
 * @return TVM runtime module wrapper
 */
TVMContribTorchRuntimeModule* tvm_contrib_torch_decode(const char* state);

/*
 * Delete DLPackTensorExt pointer.
 */
void tvm_contrib_torch_free_dlpack_tensor_ext_array(DLPackTensorExt*);

/*
 * Delete char array pointer.
 */
void tvm_contrib_torch_free_encoding(char* encoding);

/*
 * Checking if a DLPackTensorExt is boolean or cannot be copied in zero cost.
 */
bool tvm_contrib_torch_tensor_ability_of_zero_copy(DLPackTensorExt*);
}

#endif  // TVM_CONTRIB_TORCH_TVM_MODULE_WRAPPER_RUNTIME_BRIDGE_H_
