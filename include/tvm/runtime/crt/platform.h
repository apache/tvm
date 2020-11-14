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
void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t code);

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_PLATFORM_H_
