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
 * \file tvm/runtime/crt/platform.h
 * \brief The virtual memory manager for micro-controllers
 */

#ifndef TVM_RUNTIME_CRT_PLATFORM_H_
#define TVM_RUNTIME_CRT_PLATFORM_H_

#include <stdarg.h>
#include <stddef.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/error_codes.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Called when an internal error occurs and execution cannot continue.
 *
 * The platform should ideally restart or hang at this point.
 *
 * \param code An error code.
 */
#if defined(_MSC_VER)
__declspec(noreturn) void TVMPlatformAbort(tvm_crt_error_t code);
#else
void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t code);
#endif

/*! \brief Called by the microTVM RPC server to implement TVMLogf.
 *
 * Not required to be implemented when the RPC server is not linked into the binary. This
 * function's signature matches that of vsnprintf, so trivial implementations can just call
 * vsnprintf.
 *
 * \param out_buf A char buffer where the formatted string should be written.
 * \param out_buf_size_bytes Number of bytes available for writing in out_buf.
 * \param fmt The printf-style formatstring.
 * \param args extra arguments to be formatted.
 * \return number of bytes written.
 */
size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args);

/*!
 * \brief Allocate memory for use by TVM.
 *
 * When this function returns something other than kTvmErrorNoError, *out_ptr should not be modified
 * and the caller is not obligated to call TVMPlatformMemoryFree in order to avoid a memory leak.
 *
 * \param num_bytes Number of bytes requested.
 * \param dev Execution device that will be used with the allocated memory. Fixed to {kDLCPU, 0}.
 * \param out_ptr A pointer to which is written a pointer to the newly-allocated memory.
 * \return kTvmErrorNoError if successful; a descriptive error code otherwise.
 */
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr);

/*!
 * \brief Free memory used by TVM.
 *
 * \param ptr A pointer returned from TVMPlatformMemoryAllocate which should be free'd.
 * \param dev Execution device passed to TVMPlatformMemoryAllocate. Fixed to {kDLCPU, 0}.
 * \return kTvmErrorNoError if successful; a descriptive error code otherwise.
 */
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev);

/*! \brief Start a device timer.
 *
 * The device timer used must not be running.
 *
 * \return kTvmErrorNoError if successful; a descriptive error code otherwise.
 */
tvm_crt_error_t TVMPlatformTimerStart();

/*! \brief Stop the running device timer and get the elapsed time (in microseconds).
 *
 * The device timer used must be running.
 *
 * \param elapsed_time_seconds Pointer to write elapsed time into.
 *
 * \return kTvmErrorNoError if successful; a descriptive error code otherwise.
 */
tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds);

/*! \brief Platform-specific before measurement call.
 *
 * A function which is called before calling TVMFuncCall in the TimeEvaluator.
 * Can be used, for example, to initialize reset global state which may affect the results of
 * measurement.
 *
 * \return kTvmErrorNoError if successful; a descriptive error code otherwise.
 */
tvm_crt_error_t TVMPlatformBeforeMeasurement();

/*! \brief Platform-specific after measurement call.
 *
 * A function which is called after calling TVMFuncCall in the TimeEvaluator.
 * It is the counterpart of the TVMPlatformBeforeMeasurement function.
 *
 * \return kTvmErrorNoError if successful; a descriptive error code otherwise.
 */
tvm_crt_error_t TVMPlatformAfterMeasurement();

/*! \brief Fill a buffer with random data.
 *
 * Cryptographically-secure random data is NOT required. This function is intended for use
 * cases such as filling autotuning input tensors and choosing the nonce used for microTVM RPC.
 *
 * This function does not need to be implemented for inference tasks. It is used only by
 * AutoTVM and the RPC server. When not implemented, an internal weak-linked stub is provided.
 *
 * Please take care that across successive resets, this function returns different sequences of
 * values. If e.g. the random number generator is seeded with the same value, it may make it
 * difficult for a host to detect device resets during autotuning or host-driven inference.
 *
 * \param buffer Pointer to the 0th byte to write with random data. `num_bytes` of random data
 * should be written here.
 * \param num_bytes Number of bytes to write.
 * \return kTvmErrorNoError if successful; a descriptive error code otherwise.
 */
tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes);

/*! \brief Initialize TVM inference.
 *
 * Placeholder function for TVM inference initializations on a specific platform.
 * A common use of this function is setting up workspace memory for TVM inference.
 *
 * \return kTvmErrorNoError if successful.
 */
tvm_crt_error_t TVMPlatformInitialize();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_PLATFORM_H_
