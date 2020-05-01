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
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#define FARF_ERROR 1
#include "AEEStdDef.h"
#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_mem.h"
#include "HAP_perf.h"
#include "qurt.h"
#include "tvm_hvx.h"
#include "tvm_remote_nd.h"

struct msg_call {
  uint32_t func_va;
  uint32_t scalar_num;
  uint32_t stack_num;
  uint32_t data[];
} __attribute__((packed));

__attribute__((naked)) uint32_t launcher(volatile msg_call* mc,
                                         uint64_t* pcc) {
  __asm__(
      "// This function is intentionally written to be readable,      \n"
      "// rather than fast.                                           \n"
      "// r0 = value of 'volatile msg_call *mc'                       \n"
      "// r1 = address where to store the program cycle count         \n"

      "// In this packet the store happens before the allocframe so   \n"
      "// the offset added to r29 must reflect that the r29 has not   \n"
      "// yet been updated (stack grows towards decreasing addresses):\n"
      "//                    r29 before allocframe --.                \n"
      "//   [ r17:16 ] [ r19:18 ] [ r21:20 ] [ FP/LR ]                \n"
      "//   `-- r29 after allocframe      increasing addresses -->    \n"
      "{ memd(r29+#-16) = r21:20                                      \n"
      "  allocframe(#24)          }                                   \n"
      "{ memd(r29+#0) = r17:16                                        \n"
      "  memd(r29+#8) = r19:18    }                                   \n"
      "{ r17:16 = combine(r1,r0)                                      \n"
      "  r18 = r29                                                    \n"
      "  r1 = memw(r0+#4)            // scalar_num                    \n"
      "  r2 = memw(r0+#8)         }  // stack_num                     \n"
      "// If there are no stack values, skip the stack setup.         \n"
      "{ p0 = cmp.eq(r2,#0)                                           \n"
      "  if (p0.new) jump:t .Llauncher1 }                             \n"

      "// Allocate space on the stack. Let r2 = needed space          \n"
      "// rounded up to a multiple of 8.                              \n"
      "{ loop0(.Llauncher0,r2)                                        \n"
      "  r2 = asl(r2,#2)          }                                   \n"
      "{ r2 = add(r2,#4)          }                                   \n"
      "{ r2 = clrbit(r2,#2)       }                                   \n"
      "{ r29 = sub(r29,r2)        }                                   \n"

      "// Copy stack contents onto the stack. Stack contents start    \n"
      "// at r3 = r0 + offsetof(data) + scalar_num*4                  \n"
      "{ r3 = addasl(r0,r1,#2)                                        \n"
      "  r4 = r29                 }                                   \n"
      "{ r3 = add(r3,#12)         } // offsetof(data)                 \n"
      ".Llauncher0:                                                   \n"
      "{ r5 = memw(r3++#4)                                            \n"
      "  memw(r4++#4) = r5.new    } :endloop0                         \n"

      "// Load registers. Some of the loaded data may actually be     \n"
      "// values from the stack part of 'data', but it's not an issue.\n"
      ".Llauncher1:                                                   \n"
      "{ r0 = memw(r16+#12)         // mc + offsetof(data)            \n"
      "  r1 = memw(r16+#16)       }                                   \n"
      "{ r2 = memw(r16+#20)                                           \n"
      "  r3 = memw(r16+#24)       }                                   \n"
      "{ r4 = memw(r16+#28)                                           \n"
      "  r5 = memw(r16+#32)       }                                   \n"

      "// Call.                                                       \n"
      "{ r6 = memw(r16+#0)                                            \n"
      "  r21:20 = upcycle         }                                   \n"
      "{ callr r6                 }                                   \n"

      "// Restore stack pointer (free up r18), calculate cycle count. \n"
      "{ r29 = r18                                                    \n"
      "  r19:18 = upcycle         }                                   \n"
      "{ r19:18 = sub(r19:18, r21:20) }                               \n"

      "// Store pcount, restore non-volatile registers, and return.   \n"
      "{ memd(r17+#0) = r19:18                                        \n"
      "  r21:20 = memd(r29+#16)   }                                   \n"
      "{ r19:18 = memd(r29+#8)                                        \n"
      "  r17:16 = memd(r29+#0)    }                                   \n"
      "{ dealloc_return           } // implicit-use r1:0              \n");
}

extern "C" {
#pragma weak __wrap_pthread_create
int __wrap_pthread_create(pthread_t* restrict thread,
                          const pthread_attr_t* restrict attr,
                          void* (*start)(void*), void* restrict arg) {
  FARF(ERROR, "Wrong %s called", __func__);
  abort();
}
}

static void* lib_rt = nullptr;
static void* lib_thread = nullptr;

/*!
 *  \brief Perform initialization.
 *
 *  \return 0 on success, negative value on error.
 */
int tvm_remote_nd_open() {
  lib_thread = dlopen("libtvm_wrap_pthread.so", RTLD_NOW | RTLD_GLOBAL);
  if (lib_thread == nullptr) {
    FARF(ERROR, "%s: dlopen failed for libtvm_wrap_pthread.so: %s", __func__,
         dlerror());
    return AEE_EUNABLETOLOAD;
  }

  lib_rt = dlopen("libtvm_runtime.so", RTLD_NOW | RTLD_GLOBAL);
  if (lib_rt == nullptr) {
    FARF(ERROR, "%s: dlopen failed for libtvm_runtime.so: %s", __func__,
         dlerror());
    return AEE_EUNABLETOLOAD;
  }
  return AEE_SUCCESS;
}

/*!
 *  \brief Perform cleanup.
 *
 *  \return 0 on success, negative value on error.
 */
int tvm_remote_nd_close() {
  if (lib_thread != nullptr) {
    dlclose(lib_thread);
    lib_thread = nullptr;
  }
  if (lib_rt != nullptr) {
    dlclose(lib_rt);
    lib_rt = nullptr;
  }
  return AEE_SUCCESS;
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
int tvm_remote_nd_call_mmap64() {
  return AEE_SUCCESS;
}

/*!
 *  \brief  Load a shared library.
 *
 *  \param soname       Name of the shared library.
 *  \param soname_len   Length of the name.
 *  \param lib_ptr      Where to store the handle of the loaded libarary.
 *
 *  \return 0 on success, negative value on error.
 */
int tvm_remote_nd_load_library(const char* soname, int soname_len,
                               tvm_remote_nd_handle_t* lib_ptr) {
  // We need to use RTLD_NOW, the libraries we build for Hexagon
  // offloading do not support lazy binding.
  FARF(ALWAYS, "%s: %s", __func__, soname);
  if (void* lib = dlopen(soname, RTLD_GLOBAL | RTLD_NOW)) {
    *lib_ptr = reinterpret_cast<tvm_remote_nd_handle_t>(lib);
    return AEE_SUCCESS;
  }
  FARF(ERROR, "%s: dlopen failed: %s", __func__, dlerror());
  return AEE_EUNKNOWN;
}

/*!
 *  \brief  Resolve symbol name to an address.
 *
 *  \param lib          Handle of the shared library with the symbol.
 *  \param name         Symbol name.
 *  \param name_len     Length of the name.
 *  \param sym_ptr      Where to store the resolved address.
 *
 *  \return 0 on success, negative value on error.
 */
int tvm_remote_nd_get_symbol(tvm_remote_nd_handle_t lib, const char* name,
                             int name_len, tvm_remote_nd_handle_t* sym_ptr) {
  FARF(ALWAYS, "%s: name=%s", __func__, name);
  if (void* p = dlsym(reinterpret_cast<void*>(lib), name)) {
    *sym_ptr = reinterpret_cast<tvm_remote_nd_handle_t>(p);
    return AEE_SUCCESS;
  }

  FARF(ERROR, "%s: dlsym failed: %s", __func__, dlerror());
  return AEE_EUNKNOWN;
}

static void print_msg_call(const msg_call& mc) {
  FARF(ALWAYS, "device: launching %x scalar_num:%d stack_num:%d", mc.func_va,
       mc.scalar_num, mc.stack_num);
  for (unsigned i = 0; i != mc.scalar_num; ++i) {
    FARF(ALWAYS, "scalar_data[%d]  %x", i, mc.data[i]);
  }
  for (unsigned i = 0; i != mc.stack_num; ++i) {
    FARF(ALWAYS, "stack_data[%d]   %x", i, mc.data[mc.scalar_num + i]);
  }
}

/*!
 *  \brief Call the specified function.
 *
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
int tvm_remote_nd_kernel(
    tvm_remote_nd_handle_t lib, tvm_remote_nd_handle_t symbol,
    const int* scalar, int scalar_len, const int* stack, int stack_len,
    const tvm_remote_nd_buffer* scalar_in_octet, int scalar_in_octet_len,
    tvm_remote_nd_buffer* scalar_out_octet, int scalar_out_octet_len,
    const tvm_remote_nd_buffer* stack_in_octet, int stack_in_octet_len,
    tvm_remote_nd_buffer* stack_out_octet, int stack_out_octet_len,
    uint64* pcycles, uint64* time_usec) {
  hvx::config_t hvx_info = {0};
  hvx::prepare_mt_job(&hvx_info);

  int lock_result;
  // Check if HVX units are available
  if (hvx_info.num_reserved > 0) {
    lock_result = hvx::lock(hvx::MODE_128B);
    if (lock_result < 0) {
      FARF(ERROR, "%s: HVX locking failed lock_result=%d num_reserved=%d",
           __func__, lock_result, hvx_info.num_reserved);
    } else {
      FARF(ALWAYS, "%s: HVX lock successful lock_result=%d", __func__,
           lock_result);
    }
  } else {
    FARF(ERROR, "%s: there are no HVX units available", __func__);
  }

  struct msg_call* mc = (struct msg_call*)malloc(sizeof(uint32_t) *
                                                 (3 + scalar_len + stack_len));
  if (mc == nullptr) {
    FARF(ERROR, "%s: failed to allocate memory for mc", __func__);
    return AEE_ENOMEMORY;
  }

  int32_t* mc_ptr = reinterpret_cast<int32_t*>(mc);
  // Scalar buffers come first.
  int k = 3;
  for (int i = 0; i < scalar_len; i++, k++) {
    *(mc_ptr + k) = static_cast<uint32_t>(scalar[i]);
  }

  for (int i = 0; i < stack_len; i++, k++) {
    *(mc_ptr + k) = static_cast<uint32_t>(stack[i]);
  }

  mc->scalar_num = scalar_len;
  mc->stack_num = stack_len;
  mc->func_va = symbol;
  print_msg_call(*mc);
  uint64_t start_time = HAP_perf_get_time_us();
  int result = launcher(mc, pcycles);
  *time_usec = HAP_perf_get_time_us() - start_time;
  FARF(ALWAYS, "kernel execution: %llu pcycles  %llu usec", *pcycles,
       *time_usec);
  if (lock_result > 0) hvx::unlock();
  hvx::cleanup_mt_job(&hvx_info);
  if (mc) free(mc);
  return result;
}

/*!
 *  \brief Release previously loaded shared object.
 *
 *  \param lib          Handle of shared library to release.
 *
 *  \return 0 on success, negative value on error.
 */
int tvm_remote_nd_release_library(tvm_remote_nd_handle_t lib) {
  // FARF(ALWAYS, "tvm_remote_nd_release_library begin ");
  dlclose(reinterpret_cast<void*>(lib));
  FARF(ALWAYS, "tvm_remote_nd_release_library done ");
  return 0;
}
