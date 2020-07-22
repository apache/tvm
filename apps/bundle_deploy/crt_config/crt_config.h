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

/*! Support low-level debugging in MISRA-C runtime */
#define TVM_CRT_DEBUG 0

/*! Maximum supported dimension in NDArray */
#define TVM_CRT_MAX_NDIM 6
/*! Maximum supported arguments in generated functions */
#define TVM_CRT_MAX_ARGS 10
/*! Maximum supported string length in dltype, e.g. "int8", "int16", "float32" */
#define TVM_CRT_STRLEN_DLTYPE 10
/*! Maximum supported string length in function names */
#define TVM_CRT_STRLEN_NAME 80

/*!
 * \brief Log memory pool size for virtual memory allocation
 *
 * Here is a list of possible choices:
 * * use 16 for 64 KiB memory space
 * * use 17 for 128 KiB memory space
 * * use 18 for 256 KiB memory space
 * * use 19 for 512 KiB memory space
 * * use 20 for 1 MiB memory space
 * * use 21 for 2 MiB memory space
 * * use 22 for 4 MiB memory space
 * * use 23 for 8 MiB memory space
 * * use 24 for 16 MiB memory space
 * * use 25 for 32 MiB memory space
 * * use 26 for 64 MiB memory space
 * * use 27 for 128 MiB memory space
 * * use 28 for 256 MiB memory space
 */
#define TVM_CRT_LOG_VIRT_MEM_SIZE 24

/*! \brief Page size for virtual memory allocation */
#define TVM_CRT_PAGE_BYTES_LOG 12

/*! Maximum number of registered modules. */
#define TVM_CRT_MAX_REGISTERED_MODULES 2

/*! Size of the global function registry, in bytes. */
#define TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES 200

#endif  // TVM_RUNTIME_CRT_CONFIG_H_
