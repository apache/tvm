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
 * \brief TVM Executor for the Ahead-of-Time Runtime
 *
 * AOT models are described by the TVM model descriptor format
 * which can be passed to tvm_runtime_run. These descriptors will be
 * generated by the AOT compilation process. This can optionally be
 * augmented with platform specific context to be passed to the TVM
 * operators.
 *
 * Example:
 * extern tvm_model_t my_network;
 * int main() {
 *    void* data = get_data();
 *    void* output[4] = {0, 0, 0, 0};
 *    void* inputs = {data};
 *    void* outputs = {output};
 *    tvm_context_t my_context = {
 *      .driver = ...;
 *    };
 *    tvm_runtime_run(
 *      &my_network,
 *      inputs,
 *      outputs
 *      &my_context
 *    );
 *    return 0;
 * }
 */

#ifndef TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_AOT_EXECUTOR_AOT_EXECUTOR_H_
#define TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_AOT_EXECUTOR_AOT_EXECUTOR_H_

#include <stdint.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/crt/error_codes.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief TVM Model descriptor to describe the
 *  model to the runtime.
 */
typedef struct {
  size_t num_input_tensors;     /** Number of expected input tensors */
  size_t num_output_tensors;    /** Number of expected output tensors */
  TVMBackendPackedCFunc run_func; /** Generated model function, called through tvm_runtime_run */
} tvm_model_t;

/*!
 * \brief Main entry point to execute the AOT runner function
 * \param model Model descriptor structure to reference for runtime information
 * \param inputs Pointer to input pointer(s)
 * \param outputs Pointer to output pointer(s)
 * \return tvm_status_t containing success or errors from the model run
 */
tvm_crt_error_t tvm_runtime_run(const tvm_model_t* model, void** inputs, void** outputs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_AOT_EXECUTOR_AOT_EXECUTOR_H_
