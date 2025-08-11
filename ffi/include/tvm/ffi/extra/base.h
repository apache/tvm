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
 * \file tvm/ffi/extra/base.h
 * \brief Base header for Extra API.
 *
 * The extra APIs contains a minmal set of extra APIs that are not
 * required to support essential core functionality.
 */
#ifndef TVM_FFI_EXTRA_BASE_H_
#define TVM_FFI_EXTRA_BASE_H_

#include <tvm/ffi/c_api.h>

/*!
 * \brief Marks the API as extra c++ api that is defined in cc files.
 *
 * They are implemented in cc files to reduce compile-time overhead.
 * The input/output only uses POD/Any/ObjectRef for ABI stability.
 * However, these extra APIs may have an issue across MSVC/Itanium ABI,
 *
 * Related features are also available through reflection based function
 * that is fully based on C API
 *
 * The project aims to minimize the number of extra C++ APIs to keep things
 * lightweight and restrict the use to non-core functionalities.
 */
#ifndef TVM_FFI_EXTRA_CXX_API
#define TVM_FFI_EXTRA_CXX_API TVM_FFI_DLL
#endif

#endif  // TVM_FFI_EXTRA_BASE_H_
