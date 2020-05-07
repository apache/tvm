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
 * \file utvm_runtime.h
 * \brief uTVM runtime headers
 */
#ifndef TVM_RUNTIME_MICRO_HOST_DRIVEN_UTVM_RUNTIME_H_
#define TVM_RUNTIME_MICRO_HOST_DRIVEN_UTVM_RUNTIME_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>

/*!
 * \brief TODO
 */
enum UTVMReturnCode {
  UTVM_ERR_OK = 0,
  UTVM_ERR_NOT_FINISHED = -1,
  UTVM_ERR_TIMER_NOT_IMPLEMENTED = -2,
  UTVM_ERR_TIMER_OVERFLOW = -3,
  UTVM_ERR_WS_DOUBLE_FREE = -4,
  UTVM_ERR_WS_OUT_OF_SPACE = -5,
  UTVM_ERR_WS_TOO_MANY_ALLOCS = -6,
  UTVM_ERR_WS_ZERO_SIZE_ALLOC = -7,
  UTVM_ERR_WS_UNALIGNED_START = -8,
  UTVM_ERR_WS_UNALIGNED_ALLOC_SIZE = -9,
};

/*!
 * \brief Task structure for uTVM
 */
typedef struct {
  /*! \brief Pointer to function to call for this task */
  int32_t (*func)(void*, void*, int32_t);
  /*! \brief Array of argument values */
  TVMValue* arg_values;
  /*! \brief Array of type codes for each argument value */
  int* arg_type_codes;
  /*! \brief Number of arguments */
  int32_t num_args;
} UTVMTask;

/*!
 * \brief microTVM processor startup.
 * Expected to reset the stack pointer, configure any hardware required to support the CRT
 * (i.e. FPU), and then jump to UTVMMain.
 */
extern void UTVMInit();

/*!
 * \brief Start the on-device timer.
 * \return UTVMReturnCode indicating the outcome of the operation.
 */
extern int32_t UTVMTimerStart();

/*!
 * \brief Stop the on-device timer.
 * TODO(areusch): Use an SI specification of timer units here.
 * \param err Receives a UTVMReturnCode indicating the outcome of the operation.
 * \return elapsed time since UTVMTimerStart returned, in device timer ticks.
 */
extern uint32_t UTVMTimerStop(int32_t* err);

/*!
 * \brief Main entry point for UTVM runtime.
 * Waits for "go" signal, then executes tasks and reports result. Should never return.
 */
void UTVMMain();

/*!
 * \brief Function entered when UTVMMain is complete.
 * Should never return. The host sets a breakpoint here to detect end of computation.
 */
void UTVMDone();

// GCC -O3 begins to inject memset and memmove calls, so we provide impls in
// the runtime for this case and for general usage.

void *memset(void *s, int c, size_t n);

void *memmove(void *to, const void *from, size_t n);

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif

#endif  // TVM_RUNTIME_MICRO_HOST_DRIVEN_UTVM_RUNTIME_H_
