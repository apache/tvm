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
 * \brief gotvm native interface declaration.
 * \file gotvm.h
 *
 * These declarations are in cgo interface definition while calling API
 * across golang and native C boundaries.
 */

#ifndef GOTVM_GOTVM_H_
#define GOTVM_GOTVM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <dlpack/dlpack.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>

// Some type definitions for golang "C"
typedef void* native_voidp;

// Version
extern char* _TVM_VERSION(void);

// Wrappers : For incompatible cgo API.
// To handle array of strings wrapped into __gostring__
extern int _TVMFuncListGlobalNames(void*);
// To handle TVMValue slice to/from native sequential TVMValue array.
extern void _TVMValueNativeSet(void* to, void* from, int index);
extern void _TVMValueNativeGet(void* to, void* from, int index);

// Callbacks
extern int _ConvertFunction(void* fptr, void* funp);

#ifdef __cplusplus
}
#endif
#endif  // GOTVM_GOTVM_H_
