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

#ifndef TVM_RUNTIME_HEXAGON_TARGET_FASTRPC_TVM_HEXAGON_REMOTE_ND_H_
#define TVM_RUNTIME_HEXAGON_TARGET_FASTRPC_TVM_HEXAGON_REMOTE_ND_H_
/// @file tvm_hexagon_remote_nd.idl
/// IDL to offload TVM kernels to Hexagon from APPS for non-domains
#include "AEEStdDef.h"
#include "remote.h"
#ifndef __QAIC_HEADER
#define __QAIC_HEADER(ff) ff
#endif  // __QAIC_HEADER

#ifndef __QAIC_HEADER_EXPORT
#define __QAIC_HEADER_EXPORT
#endif  // __QAIC_HEADER_EXPORT

#ifndef __QAIC_HEADER_ATTRIBUTE
#define __QAIC_HEADER_ATTRIBUTE
#endif  // __QAIC_HEADER_ATTRIBUTE

#ifndef __QAIC_IMPL
#define __QAIC_IMPL(ff) ff
#endif  // __QAIC_IMPL

#ifndef __QAIC_IMPL_EXPORT
#define __QAIC_IMPL_EXPORT
#endif  // __QAIC_IMPL_EXPORT

#ifndef __QAIC_IMPL_ATTRIBUTE
#define __QAIC_IMPL_ATTRIBUTE
#endif  // __QAIC_IMPL_ATTRIBUTE
#ifdef __cplusplus
extern "C" {
#endif
typedef struct _tvm_hexagon_remote_nd_buffer__seq_octet
    _tvm_hexagon_remote_nd_buffer__seq_octet;
typedef _tvm_hexagon_remote_nd_buffer__seq_octet tvm_hexagon_remote_nd_buffer;
struct _tvm_hexagon_remote_nd_buffer__seq_octet {
  unsigned char* data;
  int dataLen;
};
typedef unsigned int tvm_hexagon_remote_nd_handle_t;
typedef uint64 tvm_hexagon_remote_nd_scalar_t;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_nd_open)(void)
    __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_nd_close)(void)
    __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_nd_load_library)(
    const char* soname, int sonameLen, const char* code, int codeLen,
    tvm_hexagon_remote_nd_handle_t* module_ptr) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_nd_get_symbol)(
    tvm_hexagon_remote_nd_handle_t module_ptr, const char* name, int nameLen,
    tvm_hexagon_remote_nd_handle_t* sym_ptr) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_nd_kernel)(
    tvm_hexagon_remote_nd_handle_t module_ptr,
    tvm_hexagon_remote_nd_handle_t symbol, int* scalar, int scalarLen,
    int* stack, int stackLen,
    const tvm_hexagon_remote_nd_buffer* scalar_in_octet,
    int scalar_in_octetLen, tvm_hexagon_remote_nd_buffer* scalar_out_octet,
    int scalar_out_octetLen,
    const tvm_hexagon_remote_nd_buffer* stack_in_octet, int stack_in_octetLen,
    tvm_hexagon_remote_nd_buffer* stack_out_octet, int stack_out_octetLen,
    uint64* pcycles, uint64* time_usec) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_nd_release_library)(
    tvm_hexagon_remote_nd_handle_t module_ptr) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_nd_call_mmap64)(void)
    __QAIC_HEADER_ATTRIBUTE;
#ifdef __cplusplus
}
#endif
#endif  // TVM_RUNTIME_HEXAGON_TARGET_FASTRPC_TVM_HEXAGON_REMOTE_ND_H_
