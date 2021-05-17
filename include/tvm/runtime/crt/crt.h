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
 * \file tvm/runtime/crt/crt.h
 * \brief Defines core life cycle functions used by CRT.
 */

#ifndef TVM_RUNTIME_CRT_CRT_H_
#define TVM_RUNTIME_CRT_CRT_H_

#include <inttypes.h>
#include <tvm/runtime/crt/error_codes.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Initialize various data structures used by the runtime.
 * Prior to calling this, any initialization needed to support TVMPlatformMemory* functions should
 * be completed.
 * \return An error code describing the outcome of initialization. Generally, initialization
 *     is only expected to fail due to a misconfiguration.
 */
tvm_crt_error_t TVMInitializeRuntime();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_CRT_H_
