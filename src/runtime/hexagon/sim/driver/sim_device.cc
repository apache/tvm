/*!
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
 * This Contribution is being provided by Qualcomm Technologies, Inc.,
 * a Delaware corporation, or its subsidiary Qualcomm Innovation Center, Inc.,
 * a California corporation, under certain additional terms and conditions
 * pursuant to Section 5 of the Apache 2.0 license.  In this regard, with
 * respect to this Contribution, the term "Work" in Section 1 of the
 * Apache 2.0 license means only the specific subdirectory within the TVM repo
 * (currently at https://github.com/dmlc/tvm) to which this Contribution is
 * made.
 * In any case, this submission is "Not a Contribution" with respect to its
 * permitted use with any of the "vta" and "verilog" subdirectories in the TVM
 * repo.
 * Qualcomm Technologies, Inc. and Qualcomm Innovation Center, Inc. retain
 * copyright of their respective Contributions.
 */

// Build with:
/*
  hexagon-clang++ -O2 sim_device.cc -o sim_dev -G0 -ldl -stdlib=libstdc++ \
                  -Wl,--force-dynamic -Wl,-E -Wl,--whole-archive -lm \
                  -Isrc/runtime/hexagon/sim
*/

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include "hexagon_sim_proto.h"

std::string time_now() {
  char str[11];  // [hh:mm:ss]
  time_t time_value = time(NULL);
  tm now;
  tm* pnow = localtime_r(&time_value, &now);

  snprintf(str, sizeof(str), "[%02d:%02d:%02d]", pnow->tm_hour, pnow->tm_min,
           pnow->tm_sec);
  return std::string(str);
}

#define LOG(FMT, ...)                                                  \
  fprintf(stderr, "%s %s:%d: " FMT "\n", time_now().c_str(), __FILE__, \
          __LINE__, ##__VA_ARGS__)

extern "C" {
// Type definition copied from include/tvm/runtime/c_backend_api.h.
typedef struct {
  /*!
   * \brief Auxiliary used for synchronization
   */
  void* sync_handle;
  /*! \brief total amount of task */
  int32_t num_task;
} TVMParallelGroupEnv;

/*!
 * \brief The callback function to execute a parallel lambda
 * \param task_id the task id of the function.
 * \param penv The parallel environment backs the execution.
 * \param cdata The supporting closure data.
 */
typedef int (*FTVMParallelLambda)(int task_id, TVMParallelGroupEnv* penv,
                                  void* cdata);

int TVMBackendParallelLaunch(FTVMParallelLambda kernel, void* cdata,
                             int num_task);
int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv);
}

/*!
 * The implementation of the parallel runtime for execution on simulator.
 * The simulator environment does not support running multiple threads,
 * so the runtime is trivial: the maximum number of threads that it
 * supports it 1 (i.e. the main thread only).
 */
int TVMBackendParallelLaunch(FTVMParallelLambda kernel, void* cdata,
                             int num_task) {
  TVMParallelGroupEnv penv{nullptr, 1};
  LOG("%s(kernel=%p, cdata=%p, num_task=%d)", __func__, kernel, cdata,
      num_task);
  kernel(0, &penv, cdata);
  return 0;
}

int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv) {
  LOG("%s(task_id=%d, penv=%p)", __func__, task_id, penv);
  assert(task_id == 0 && penv->num_task == 1 && "Expecting single task");
  return 0;
}

struct allocator {
 private:
  struct block {
    block(void* p, size_t s) : ptr_(p), size_(s) {}
    bool operator<(const block& b) const {
      return uintptr_t(ptr_) < uintptr_t(b.ptr_);
    }
    void* ptr_;
    size_t size_;
  };

  using vector_type = std::vector<block>;
  using iterator = vector_type::iterator;
  vector_type allocations_;

 public:
  void* alloc(unsigned size, size_t align);
  void free(void* p);
};

void* allocator::alloc(unsigned size, size_t align) {
  void* ptr = aligned_alloc(align, size);
  if (ptr == nullptr) {
    perror("device: error allocating memory:");
    return ptr;
  }

  block b(ptr, size);
  iterator i = std::lower_bound(allocations_.begin(), allocations_.end(), b);
  iterator w = allocations_.insert(i, b);
  if (w != allocations_.begin()) {
    iterator pw = w - 1;
    assert(uintptr_t(pw->ptr_) + pw->size_ < uintptr_t(w->ptr_));
  }
  if (w + 1 != allocations_.end()) {
    iterator nw = w + 1;
    assert(uintptr_t(w->ptr_) + w->size_ <= uintptr_t(nw->ptr_));
  }

  LOG("device: allocated %d bytes aligned at %d: %p", size, align, ptr);
  return ptr;
}

void allocator::free(void* ptr) {
  LOG("device: freeing %p", ptr);
  iterator i = std::lower_bound(allocations_.begin(), allocations_.end(),
                                block(ptr, 0));
  assert(i != allocations_.end());
  assert(i->ptr_ == ptr);
  ::free(i->ptr_);
  allocations_.erase(i);
}

static void print_msg_call(const MsgCall& mc) {
  auto to_dec_string = [](int v) {
    char tmp[11];
    snprintf(tmp, sizeof(tmp), "%d", v);
    return std::string(tmp);
  };
  auto to_hex_string = [](uint32_t v) {
    char tmp[9];
    snprintf(tmp, sizeof(tmp), "%lx", v);
    return std::string(tmp);
  };
  std::string str = "device: launching " + to_hex_string(mc.func_va) +
                    " sc:" + to_dec_string(mc.scalar_num) + " {";
  for (unsigned i = 0; i != mc.scalar_num; ++i) {
    str += ' ' + to_hex_string(mc.data[i]);
    if (i + 1 != mc.scalar_num) str += ',';
  }
  str += " }, st:" + to_dec_string(mc.stack_num) + " {";
  for (unsigned i = 0; i != mc.stack_num; ++i) {
    str += ' ' + to_hex_string(mc.data[i + mc.scalar_num]);
    if (i + 1 != mc.stack_num) str += ',';
  }
  str += " }";
  LOG("%s", str.c_str());
}

static std::vector<MsgCall*> task_queue;

struct environment {
  allocator alloc;
  void* dl_handle = nullptr;
};

extern "C" {
volatile Message message_buffer;
int dispatch(environment* env) __attribute__((noinline));
}

static volatile unsigned char payload_buffer[4096];

void set_msg(uint32_t code, uint32_t len, uint32_t va) {
  message_buffer.code = code;
  message_buffer.len = len;
  message_buffer.va = va;
}

inline void* pointer(uint32_t v) {
  return reinterpret_cast<void*>(static_cast<uintptr_t>(v));
}

inline uint32_t va(const volatile void* p) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}

__attribute__((naked)) uint32_t launcher(volatile MsgCall* mc, uint64_t* pcc) {
  __asm__(
      "// This function is intentionally written to be readable,      \n"
      "// rather than fast.                                           \n"
      "// r0 = value of 'volatile MsgCall *mc'                       \n"
      "// r1 = address where to store the program cycle count         \n"
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

int dispatch(environment* env) {
  uint32_t code = message_buffer.code;
  // Special handling of MsgReq.
  if (code == kMsgReq) {
    // XXX: Enable fprintfs for MsqReg under #define.
    // LOG("device: {MsgReq, %lu, %lx}", message_buffer.len,
    //     message_buffer.va);
    // XXX: Implement handling of longer messages.
    assert(message_buffer.len <= sizeof(payload_buffer));
    set_msg(kMsgAck, sizeof(payload_buffer), va(payload_buffer));
    return 0;
  }

  switch (code) {
    case kAlloc: {
      LOG("device: {kAlloc, %lu, %lx}", message_buffer.len, message_buffer.va);
      assert(message_buffer.len == sizeof(MsgAlloc));
      auto* ma = reinterpret_cast<volatile MsgAlloc*>(message_buffer.va);
      void* p = env->alloc.alloc(ma->size, ma->align);
      reinterpret_cast<volatile MsgPointer*>(payload_buffer)->va = va(p);
      set_msg(kNone, sizeof(MsgPointer), va(payload_buffer));
      break;
    }
    case kFree: {
      LOG("device: {kFree, %lu, %lx}", message_buffer.len, message_buffer.va);
      assert(message_buffer.len == sizeof(MsgPointer));
      auto* mp = reinterpret_cast<volatile MsgPointer*>(message_buffer.va);
      env->alloc.free(pointer(mp->va));
      set_msg(kNone, 0u, 0u);
      break;
    }
    case kCopy: {
      LOG("device: {kCopy, %lu, %lx}", message_buffer.len, message_buffer.va);
      assert(message_buffer.len == sizeof(MsgCopy));
      auto* mc = reinterpret_cast<volatile MsgCopy*>(message_buffer.va);
      memcpy(pointer(mc->dst), pointer(mc->src), mc->len);
      set_msg(kNone, 0u, 0u);
      break;
    }
    case kLoad: {
      // LOG("device: {kLoad, %lu, %lx}",
      //     message_buffer.len, message_buffer.va);
      if (env->dl_handle != nullptr) dlclose(env->dl_handle);
      const char* name = static_cast<const char*>(pointer(message_buffer.va));
      // LOG(stderr, "device: dlopen(%s)", name);
      env->dl_handle = dlopen(name, RTLD_LAZY);
      if (env->dl_handle == nullptr) LOG("dlopen: %s\n", dlerror());
      assert(env->dl_handle != nullptr);
      reinterpret_cast<volatile MsgPointer*>(payload_buffer)->va =
          va(env->dl_handle);
      set_msg(kNone, sizeof(MsgPointer), va(payload_buffer));
      break;
    }
    case kUnload: {
      // LOG("device: {kUnload, %lu, %lx}",
      //     message_buffer.len, message_buffer.va);
      assert(env->dl_handle != nullptr);
      assert(message_buffer.len == sizeof(MsgPointer));
      auto* mp = reinterpret_cast<volatile MsgPointer*>(message_buffer.va);
      assert(pointer(mp->va) == env->dl_handle);
      dlclose(env->dl_handle);
      env->dl_handle = nullptr;
      set_msg(kNone, 0u, 0u);
      break;
    }
    case kResolve: {
      LOG("device: {kResolve, %lu, %lx}", message_buffer.len,
          message_buffer.va);
      assert(env->dl_handle != nullptr);
      dlerror();
      const char* name = static_cast<const char*>(pointer(message_buffer.va));
      void* s = dlsym(env->dl_handle, name);
      reinterpret_cast<volatile MsgPointer*>(payload_buffer)->va = va(s);
      set_msg(kNone, sizeof(MsgPointer), va(payload_buffer));
      break;
    }
    case kCall: {
      LOG("device: {kCall, %lu, %lx}", message_buffer.len, message_buffer.va);
      // Add the task to the queue.
      auto* mc = reinterpret_cast<MsgCall*>(message_buffer.va);
      uint32_t size = 4 * (3 + mc->scalar_num + mc->stack_num);
      MsgCall* t = static_cast<MsgCall*>(malloc(size));
      memcpy(t, mc, size);
      task_queue.push_back(t);
      // Return 0.
      *reinterpret_cast<volatile uint32_t*>(payload_buffer) = 0;
      set_msg(kNone, sizeof(uint32_t), va(payload_buffer));
      break;
    }
    case kFlush: {
      LOG("device: {kFlush}");
      LOG("device: %d tasks in the queue", task_queue.size());
      // Execute all tasks from the queue and release memory buffers
      // for as long as the return values are 0. Upon receiving a non-zero
      // return value, continue freeing memory but no longer execute
      // any tasks. The task queue will be cleared in any case.
      uint32_t rv = 0;
      uint64_t pcc;  // Pcycle counter, will be 0 under simulator (upcycle).
      for (MsgCall* t : task_queue) {
        if (rv == 0) {
          print_msg_call(*t);
          rv = launcher(t, &pcc);
        }
        free(t);
      }
      task_queue.clear();
      *reinterpret_cast<volatile uint32_t*>(payload_buffer) = rv;
      set_msg(kNone, sizeof(uint32_t), va(payload_buffer));
      break;
    }
    default:
      LOG("device: unknown code: %lu", message_buffer.code);
      abort();
      break;
  }
  return 0;
}

extern "C" {
int acquire_vector_unit(int);
void release_vector_unit();
}

int main() {
  environment env;
  acquire_vector_unit(0);

  const char* builtin[] = {"libgcc.so", "libc.so"};
  dlinit(2, const_cast<char**>(builtin));

  while (!dispatch(&env)) {
  }

  release_vector_unit();
  return 0;
}
