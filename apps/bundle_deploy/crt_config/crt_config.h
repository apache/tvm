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
 * \file apps/bundle_deploy/crt_config.h
 * \brief CRT configuration for bundle_deploy app.
 */
#ifndef TVM_RUNTIME_CRT_CONFIG_H_
#define TVM_RUNTIME_CRT_CONFIG_H_

/*! Log level of the CRT runtime */
#define TVM_CRT_LOG_LEVEL TVM_CRT_LOG_LEVEL_DEBUG

/*! Support low-level debugging in MISRA-C runtime */
#define TVM_CRT_DEBUG 0

/*! Maximum supported dimension in NDArray */
#define TVM_CRT_MAX_NDIM 6
/*! Maximum supported arguments in generated functions */
#define TVM_CRT_MAX_ARGS 10
/*! Maximum supported string length in dltype, e.g. "int8", "int16", "float32" */
#define TVM_CRT_MAX_STRLEN_DLTYPE 10
/*! Maximum supported string length in function names */
#define TVM_CRT_MAX_STRLEN_FUNCTION_NAME 80

/*! Maximum number of registered modules. */
#define TVM_CRT_MAX_REGISTERED_MODULES 2

/*! Size of the global function registry, in bytes. */
#define TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES 512

/*! Maximum packet size, in bytes, including the length header. */
#define TVM_CRT_MAX_PACKET_SIZE_BYTES 512

#endif  // TVM_RUNTIME_CRT_CONFIG_H_
