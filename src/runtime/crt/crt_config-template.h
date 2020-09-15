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
 * \file tvm/runtime/crt_config.h.template
 * \brief Template for CRT configuration, to be modified on each target.
 */
#ifndef TVM_RUNTIME_CRT_CRT_CONFIG_TEMPLATE_H_
#define TVM_RUNTIME_CRT_CRT_CONFIG_TEMPLATE_H_

/*! Maximum supported dimension in NDArray */
#define TVM_CRT_MAX_NDIM 6

/*! Maximum supported arguments in generated functions */
#define TVM_CRT_MAX_ARGS 10

/*! Size of the global function registry, in bytes. */
#define TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES 200

/*! Maximum number of registered modules. */
#define TVM_CRT_MAX_REGISTERED_MODULES 2

/*! Maximum packet size, in bytes, including the length header. */
#define TVM_CRT_MAX_PACKET_SIZE_BYTES 2048

/*! Maximum supported string length in dltype, e.g. "int8", "int16", "float32" */
#define TVM_CRT_MAX_STRLEN_DLTYPE 10

/*! Maximum supported string length in function names */
#define TVM_CRT_MAX_STRLEN_FUNCTION_NAME 80

/*! \brief Maximum length of a PackedFunc function name. */
#define TVM_CRT_MAX_FUNCTION_NAME_LENGTH_BYTES 30

/*! \brief DLDataType for the return value from strlen */
#define TVM_CRT_STRLEN_DLTYPE 10

#endif  // TVM_RUNTIME_CRT_CRT_CONFIG_TEMPLATE_H_
