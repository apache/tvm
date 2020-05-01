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

#include <assert.h>
#include <stdlib.h>

#define FARF_ERROR 1
#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "apps_mem.h"
#include "qurt.h"
#include "tvm_remote.h"
#include "tvm_remote_nd.h"

#if __HEXAGON_ARCH__ >= 65
#include "HAP_vtcm_mgr.h"
#else
// Stub functions for targets that don't support VTCM.
static void* HAP_request_VTCM(int a, int b) { return 0; }
static int HAP_release_VTCM(void* a) { return 0; }
static int HAP_query_avail_VTCM(unsigned* avail_block_size,
                                unsigned* max_page_size, unsigned* num_pages) {
  FARF(ALWAYS, "%s: running on architecture V62 or less", __func__);
  return AEE_ENOMEMORY;
}
#endif  // __HEXAGON_ARCH__

#define MIN_GATHER_SCATTER_SZ (32 * 1024)
#define MAX_GATHER_SCATTER_SZ (64 * 1024)
#define MIN_VTCM_SZ (64 * 1024)

/*!
 *  \brief Open a domain channel.
 *
 *  \param uri          URI of the channel description.
 *  \param handle_ptr   Where to store the channel handle.
 *
 *  \return 0 on success, negative value on error.
 */
int tvm_remote_open(const char* uri, remote_handle64* handle_ptr) {
  FARF(ALWAYS, "%s, uri=%s", __func__, uri);
  int rc = tvm_remote_nd_open();
  if (rc != AEE_SUCCESS) {
    FARF(ERROR, "%s: tvm_remote_nd_open failed rc=%08x", __func__, rc);
    return rc;
  }

  *handle_ptr =
      static_cast<remote_handle64>(reinterpret_cast<uintptr_t>(malloc(1)));
  if (!*handle_ptr) {
    FARF(ERROR, "%s: cannot allocate memory", __func__);
    return AEE_ENOMEMORY;
  }
  return AEE_SUCCESS;
}

/*!
 *  \brief Close domain channel.
 *
 *  \param handle   Domain channel handle to close.
 *
 *  \return 0 on success, negative value on error.
 */
int tvm_remote_close(remote_handle64 handle) {
  FARF(ALWAYS, "%s", __func__);
  if (handle) free(reinterpret_cast<void*>(static_cast<uintptr_t>(handle)));
  int rc = tvm_remote_nd_close();
  if (rc != AEE_SUCCESS) {
    FARF(ERROR, "%s: tvm_remote_nd_close failed rc=%08x", __func__, rc);
  }
  return rc;
}

/*!
 *  \brief Dummy function.
 *
 *  \param handle   Domain channel handle.
 *
 *  \return This function always returns 0.
 *
 * This function is present as a workaround. See comment at the call site
 * in hexagon_device_target.cc.
 */
int tvm_remote_call_mmap64(remote_handle64 handle) {
  return AEE_SUCCESS;
}

/*!
 *  \brief  Load a shared library.
 *
 *  \param handle       Domain channel handle.
 *  \param soname       Name of the shared library.
 *  \param soname_len   Length of the name.
 *  \param lib_ptr      Where to store the handle of the loaded libarary.
 *
 *  \return 0 on success, negative value on error.
 */
int tvm_remote_load_library(remote_handle64 handle, const char* soname,
                            int soname_len, tvm_remote_handle_t* lib_ptr) {
  return tvm_remote_nd_load_library(soname, soname_len, lib_ptr);
}

/*!
 *  \brief  Resolve symbol name to an address.
 *
 *  \param handle       Domain channel handle.
 *  \param lib          Handle of the shared library with the symbol.
 *  \param name         Symbol name.
 *  \param name_len     Length of the name.
 *  \param sym_ptr      Where to store the resolved address.
 *
 *  \return 0 on success, negative value on error.
 */
int tvm_remote_get_symbol(remote_handle64 handle, tvm_remote_handle_t lib,
                          const char* name, int name_len,
                          tvm_remote_handle_t* sym_ptr) {
  return tvm_remote_nd_get_symbol(lib, name, name_len, sym_ptr);
}

/*!
 *  \brief Call the specified function.
 *
 *  \param handle                 Domain channel handle.
 *  \param lib                    Handle of the library containing
 *                                the function to call.
 *  \param symbol                 Address of the function to call.
 *  \param scalar                 Address of values to pass in registers.
 *  \param scalar_len             Number of values to pass in registers.
 *  \param stack                  Address of values to pass on stack.
 *  \param stack_len              Number of values to pass on stack.
 *
 *  \param scalar_in_octet        Address of the incoming scalar buffer.
 *  \param scalar_in_octet_len    Length of the incoming scalar buffer.
 *  \param scalar_out_octet       Address of the outgoing scalar buffer.
 *  \param scalar_out_octet_len   Length of the outgoing scalar buffer.
 *  \param stack_in_octet         Address of the incoming stack buffer.
 *  \param stack_in_octet_len     Length of the incoming stack buffer.
 *  \param stack_out_octet        Address of the outgoing stack buffer.
 *  \param stack_out_octet_len    Length of the outgoing stack buffer.
 *
 *  \param pcycles                Pointer to where to store cycle count.
 *  \param time_usec              Pointer to where to store time in usec.
 *
 *  \return 0 on success, negative value on error.
 *
 * The 8 "octet" arguments in this function are used for cache operations
 * only. They are not used for procesing.
 */
int tvm_remote_kernel(
    remote_handle64 handle, tvm_remote_handle_t lib,
    tvm_remote_handle_t symbol, const int* scalar, int scalar_len,
    const int* stack, int stack_len, const tvm_remote_buffer* scalar_in_octet,
    int scalar_in_octet_len, tvm_remote_buffer* scalar_out_octet,
    int scalar_out_octet_len, const tvm_remote_buffer* stack_in_octet,
    int stack_in_octet_len, tvm_remote_buffer* stack_out_octet,
    int stack_out_octet_len, uint64* pcycles, uint64* time_usec) {
  return tvm_remote_nd_kernel(
      lib, symbol, scalar, scalar_len, stack, stack_len,
      reinterpret_cast<const tvm_remote_nd_buffer*>(scalar_in_octet),
      scalar_in_octet_len,
      reinterpret_cast<tvm_remote_nd_buffer*>(scalar_out_octet),
      scalar_out_octet_len,
      reinterpret_cast<const tvm_remote_nd_buffer*>(stack_in_octet),
      stack_in_octet_len,
      reinterpret_cast<tvm_remote_nd_buffer*>(stack_out_octet),
      stack_out_octet_len, pcycles, time_usec);
}

/*!
 *  \brief Release previously loaded shared object.
 *
 *  \param handle       Domain channel handle.
 *  \param lib          Handle of shared library to release.
 *
 *  \return 0 on success, negative value on error.
 */
int tvm_remote_release_library(remote_handle64 handle,
                               tvm_remote_handle_t lib) {
  // FARF(ALWAYS, "tvm_remote_release_library begin ");
  return tvm_remote_nd_release_library(lib);
}

/*!
 *  \brief Allocate VTCM memory.
 *
 *  \param handle   Domain channel handle.
 *  \param size     Number of bytes to allocate.
 *  \param align    Requested alignment.
 *  \param dsp_va   Address of variable to store the allocated VTCM
 *                  address to.
 *
 *  \return 0 on success, negative value on error.
 */
int tvm_remote_alloc_vtcm(remote_handle64 handle, unsigned size,
                          unsigned align, unsigned* dsp_va) {
  FARF(ALWAYS, "%s: size=%u, align=%u", __func__, size, align);
  unsigned avail_block_size, max_page_size, num_pages;
  int rc = HAP_query_avail_VTCM(&avail_block_size, &max_page_size, &num_pages);
  if (rc != AEE_SUCCESS) {
    FARF(ERROR, "%s: HAP_query_avail_VTCM failed, rc=%08x", __func__, rc);
    return rc;
  }
  FARF(ALWAYS, "%s: avail_block_size=%u, max_page_size=%u, num_pages=%u",
       __func__, avail_block_size, max_page_size, num_pages);

  if (max_page_size < MIN_VTCM_SZ) {
    FARF(ERROR, "%s: available VTCM size less than %d KB, aborting", __func__,
         MIN_VTCM_SZ / 1024);
    return AEE_ENOMEMORY;
  }

  void* vtcm_base = HAP_request_VTCM(size, /*single_page_flag=*/1);
  if (!vtcm_base) {
    FARF(ERROR, "%s: error allocating VTCM", __func__);
    return AEE_ENOMEMORY;
  }
  *dsp_va = static_cast<unsigned>(reinterpret_cast<uintptr_t>(vtcm_base));
  FARF(ALWAYS, "%s: allocated VTCM addr=0x%p", __func__, vtcm_base);
  return AEE_SUCCESS;
}

/*!
 *  \brief Free VTCM memory.
 *
 *  \param handle   Domain channel handle.
 *  \param dsp_va   VTCM address to free.
 *
 *  \return 0 on success, negative value on error.
 */
int tvm_remote_free_vtcm(remote_handle64 handle, unsigned dsp_va) {
  FARF(ALWAYS, "%s: dsp_va=0x%08x", __func__, dsp_va);
  void* vtcm_base = reinterpret_cast<void*>(dsp_va);
  int rc = HAP_release_VTCM(vtcm_base);
  if (rc != AEE_SUCCESS) {
    FARF(ERROR, "%s: error freeing VTCM, rc=%08x", __func__, rc);
  }
  return rc;
}
