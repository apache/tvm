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

#ifndef TVM_RUNTIME_HEXAGON_TARGET_FASTRPC_TVM_HEXAGON_REMOTE_H_
#define TVM_RUNTIME_HEXAGON_TARGET_FASTRPC_TVM_HEXAGON_REMOTE_H_
/// @file tvm_hexagon_remote.idl
/// IDL to offload TVM kernels to Hexagon from APPS for multi-domains
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
/**
 * Opens the handle in the specified domain.  If this is the first
 * handle, this creates the session.  Typically this means opening
 * the device, aka open("/dev/adsprpc-smd"), then calling ioctl
 * device APIs to create a PD on the DSP to execute our code in,
 * then asking that PD to dlopen the .so and dlsym the skel function.
 *
 * @param uri, <interface>_URI"&_dom=aDSP"
 *    <interface>_URI is a QAIC generated uri, or
 *    "file:///<sofilename>?<interface>_skel_handle_invoke&_modver=1.0"
 *    If the _dom parameter is not present, _dom=DEFAULT is assumed
 *    but not forwarded.
 *    Reserved uri keys:
 *      [0]: first unamed argument is the skel invoke function
 *      _dom: execution domain name, _dom=mDSP/aDSP/DEFAULT
 *      _modver: module version, _modver=1.0
 *      _*: any other key name starting with an _ is reserved
 *    Unknown uri keys/values are forwarded as is.
 * @param h, resulting handle
 * @retval, 0 on success
 */
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_open)(
    const char* uri, remote_handle64* h) __QAIC_HEADER_ATTRIBUTE;
/**
    * Closes a handle.  If this is the last handle to close, the session
    * is closed as well, releasing all the allocated resources.

    * @param h, the handle to close
    * @retval, 0 on success, should always succeed
    */
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_close)(
    remote_handle64 h) __QAIC_HEADER_ATTRIBUTE;
typedef struct _tvm_hexagon_remote_buffer__seq_octet
    _tvm_hexagon_remote_buffer__seq_octet;
typedef _tvm_hexagon_remote_buffer__seq_octet tvm_hexagon_remote_buffer;
struct _tvm_hexagon_remote_buffer__seq_octet {
  unsigned char* data;
  int dataLen;
};
typedef unsigned int tvm_hexagon_remote_handle_t;
typedef uint64 tvm_hexagon_remote_scalar_t;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_load_library)(
    remote_handle64 _h, const char* soname, int sonameLen, const char* code,
    int codeLen,
    tvm_hexagon_remote_handle_t* module_ptr) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_get_symbol)(
    remote_handle64 _h, tvm_hexagon_remote_handle_t module_ptr,
    const char* name, int nameLen,
    tvm_hexagon_remote_handle_t* sym_ptr) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_kernel)(
    remote_handle64 _h, tvm_hexagon_remote_handle_t module_ptr,
    tvm_hexagon_remote_handle_t symbol, int* scalar, int scalarLen, int* stack,
    int stackLen, const tvm_hexagon_remote_buffer* scalar_in_octet,
    int scalar_in_octetLen, tvm_hexagon_remote_buffer* scalar_out_octet,
    int scalar_out_octetLen, const tvm_hexagon_remote_buffer* stack_in_octet,
    int stack_in_octetLen, tvm_hexagon_remote_buffer* stack_out_octet,
    int stack_out_octetLen, uint64* pcycles,
    uint64* time_usec) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_release_library)(
    remote_handle64 _h,
    tvm_hexagon_remote_handle_t module_ptr) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_alloc_vtcm)(
    remote_handle64 _h, unsigned int size, unsigned int align,
    unsigned int* dsp_va) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_free_vtcm)(
    remote_handle64 _h, unsigned int dsp_va) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(tvm_hexagon_remote_call_mmap64)(
    remote_handle64 _h) __QAIC_HEADER_ATTRIBUTE;
#ifndef tvm_hexagon_remote_URI
#define tvm_hexagon_remote_URI                                            \
  "file:///"                                                              \
  "libtvm_hexagon_remote_skel.so?tvm_hexagon_remote_skel_handle_invoke&_" \
  "modver=1.0"
#endif /*tvm_hexagon_remote_URI*/
#ifdef __cplusplus
}
#endif
#endif  // TVM_RUNTIME_HEXAGON_TARGET_FASTRPC_TVM_HEXAGON_REMOTE_H_
