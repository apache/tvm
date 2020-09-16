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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_PLATFORM_H_
