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

/*
  Required options:
    -ldl -G0                  For dlinit/dlopen/dlclose.
    -Wl,--force-dynamic       Make this a dynamic executable (with dynamic
                              symbol table).
    -Wl,-E                    Export all defined symbols as dynamic.
    -Wl,--whole-archive       Link the entire contents of libc.
    -mhvx -mhvx-length=128b   Enable HVX.
    -Wno-format               Silence format warning (unsigned vs uint32_t).
*/

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include "hexagon_sim_proto.h"
#include "pthread.h"
#include "tvm/runtime/c_runtime_api.h"

static std::string timeNow() {
  char str[11];  // [hh:mm:ss]
  time_t time_value = time(NULL);
  tm* pnow = localtime(&time_value);  // NOLINT(runtime/threadsafe_fn)

  snprintf(str, sizeof(str), "[%02d:%02d:%02d]", pnow->tm_hour, pnow->tm_min, pnow->tm_sec);
  return std::string(str);
}

#define LOG(FMT, ...) \
  fprintf(stderr, "%s %s:%d: " FMT "\n", timeNow().c_str(), __FILE__, __LINE__, ##__VA_ARGS__)

using HVX_Vector = int __attribute__((__vector_size__(128))) __attribute__((aligned(128)));

static unsigned getVectorLength() {
  HVX_Vector v = __builtin_HEXAGON_V6_lvsplatw_128B(0x01010101);
  unsigned char* p = reinterpret_cast<unsigned char*>(&v);
  if (p[127] == 1) return 128;
  assert(p[63] == 1);
  return 64;
}

extern "C" {
// Print vector functions. They can be used to help debug tensorized
// code, via
// ib.emit(tvm.call_extern('int32', 'V6_pv8', 'vector:', v))
// ib.emit(tvm.call_extern('int32', 'V6_pv16', 'info:', v))
// ib.emit(tvm.call_extern('int32', 'V6_pv32', 'value:', v))

// The first argument is a string printed before the vector contents.
int V6_pv8(const char* s, HVX_Vector v);
int V6_pv16(const char* s, HVX_Vector v);
int V6_pv32(const char* s, HVX_Vector v);
}

int V6_pv8(const char* s, HVX_Vector v) {
  unsigned vlen = getVectorLength();
  uint8_t* ptr = reinterpret_cast<uint8_t*>(&v);
  fprintf(stderr, "%s:", s);
  for (unsigned i = 0; i != vlen; ++i) {
    fprintf(stderr, " %02x", ptr[i]);
  }
  fprintf(stderr, "\n");
  return 0;
}

int V6_pv16(const char* s, HVX_Vector v) {
  unsigned vlen = getVectorLength();
  uint16_t* ptr = reinterpret_cast<uint16_t*>(&v);
  fprintf(stderr, "%s:", s);
  for (unsigned i = 0; i != vlen / sizeof(uint16_t); ++i) {
    fprintf(stderr, " %04x", ptr[i]);
  }
  fprintf(stderr, "\n");
  return 0;
}

int V6_pv32(const char* s, HVX_Vector v) {
  unsigned vlen = getVectorLength();
  uint32_t* ptr = reinterpret_cast<uint32_t*>(&v);
  fprintf(stderr, "%s:", s);
  for (unsigned i = 0; i != vlen / sizeof(uint32_t); ++i) {
    fprintf(stderr, " %08x", ptr[i]);
  }
  fprintf(stderr, "\n");
  return 0;
}

extern "C" {
// Function referenced from libc++.a, but not defined in libc.a.
int clock_gettime(clockid_t clock_id, struct timespec* tp);
// pthread_create is wrapped so that we can set a bigger stack size
// for QuRT. Here this isn't needed, but we still need to implement
// the wrapper.
int __wrap_pthread_create(pthread_t* thread, const pthread_attr_t* attr,
                          void* (*start_routine)(void*), void* arg);
}

int clock_gettime(clockid_t clock_id, struct timespec* tp) {
  // Stub implementation.
  return 0;
}

int __wrap_pthread_create(pthread_t* thread, const pthread_attr_t* attr,
                          void* (*start_routine)(void*), void* arg) {
  LOG("%s", __func__);
  return pthread_create(thread, attr, start_routine, arg);
}

// FIXME(kparzysz-quic): query the cfg register to compute the VTCM base.
// This works now.
const unsigned int TCM_BASE = 0xD8000000;
const unsigned int VTCM_BASE = TCM_BASE + 0x400000;

class Allocator {
 private:
  struct Block {
    Block(void* p, size_t s) : ptr_(p), size_(s), vtcm_(false) {}
    Block(void* p, size_t s, bool v) : ptr_(p), size_(s), vtcm_(v) {}
    bool operator<(const Block& b) const { return uintptr_t(ptr_) < uintptr_t(b.ptr_); }
    void* ptr_;
    size_t size_;
    bool vtcm_;
  };

  using vector_type = std::vector<Block>;
  using iterator = vector_type::iterator;
  vector_type allocations_;

  uintptr_t cur_vtcm = VTCM_BASE;

 public:
  void* alloc(unsigned size, size_t align);
  void* vtcm_alloc(unsigned size, size_t align);
  void free(void* p);
};

void* Allocator::alloc(unsigned size, size_t align) {
  void* ptr = aligned_alloc(align, size);
  if (ptr == nullptr) {
    perror("device: error allocating memory:");
    return ptr;
  }

  Block b(ptr, size);
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

// For now, just allocation sequentially. This needs to be improved to use a
// free list.
void* Allocator::vtcm_alloc(unsigned size, size_t align) {
  uintptr_t a = cur_vtcm;
  a = (a + (align - 1)) & -align;
  cur_vtcm = a + size;
  void* ptr = reinterpret_cast<void*>(a);
  if (ptr == nullptr) {
    perror("device: error allocating vtcm memory:");
    return ptr;
  }

  Block b(ptr, size, true);
  iterator i = std::lower_bound(allocations_.begin(), allocations_.end(), b);
  iterator w = allocations_.insert(i, b);
  if (w != allocations_.begin()) {
    iterator pw = w - 1;
    assert(uintptr_t(pw->ptr_) + pw->size_ <= uintptr_t(w->ptr_));
  }
  if (w + 1 != allocations_.end()) {
    iterator nw = w + 1;
    assert(uintptr_t(w->ptr_) + w->size_ <= uintptr_t(nw->ptr_));
  }

  LOG("device: allocated vtcm %d bytes aligned at %d: %p", size, align, ptr);
  return ptr;
}

void Allocator::free(void* ptr) {
  LOG("device: freeing %p", ptr);
  iterator i = std::lower_bound(allocations_.begin(), allocations_.end(), Block(ptr, 0));
  assert(i != allocations_.end());
  assert(i->ptr_ == ptr);
  if (!i->vtcm_) ::free(i->ptr_);
  allocations_.erase(i);
}

static void printMsgCall(const MsgCall& mc) {
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

struct Environment {
  Allocator alloc;
  void* dl_handle = nullptr;
};

extern "C" {
volatile Message message_buffer;
int dispatch(Environment* env) __attribute__((noinline));
}

static volatile unsigned char payload_buffer[4096];

static void setMsg(uint32_t code, uint32_t len, uint32_t va) {
  message_buffer.code = code;
  message_buffer.len = len;
  message_buffer.va = va;
}

inline void* pointer(uint32_t v) { return reinterpret_cast<void*>(static_cast<uintptr_t>(v)); }

inline uint32_t va(const volatile void* p) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}

__attribute__((naked)) uint32_t launcher(volatile MsgCall* mc, uint64_t* pcc) {
  __asm__(
      "// This function is intentionally written to be readable,      \n"
      "// rather than fast.                                           \n"
      "// r0 = value of 'volatile MsgCall *mc'                        \n"
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

int dispatch(Environment* env) {
  uint32_t code = message_buffer.code;
  // Special handling of MsgReq.
  if (code == kMsgReq) {
    assert(message_buffer.len <= sizeof(payload_buffer));
    setMsg(kMsgAck, sizeof(payload_buffer), va(payload_buffer));
    return 0;
  }

  switch (code) {
    case kAlloc: {
      LOG("device: {kAlloc, %lu, %lx}", message_buffer.len, message_buffer.va);
      assert(message_buffer.len == sizeof(MsgAlloc));
      auto* ma = reinterpret_cast<volatile MsgAlloc*>(message_buffer.va);
      void* p = env->alloc.alloc(ma->size, ma->align);
      reinterpret_cast<volatile MsgPointer*>(payload_buffer)->va = va(p);
      setMsg(kNone, sizeof(MsgPointer), va(payload_buffer));
      break;
    }
    case kFree: {
      LOG("device: {kFree, %lu, %lx}", message_buffer.len, message_buffer.va);
      assert(message_buffer.len == sizeof(MsgPointer));
      auto* mp = reinterpret_cast<volatile MsgPointer*>(message_buffer.va);
      env->alloc.free(pointer(mp->va));
      setMsg(kNone, 0u, 0u);
      break;
    }
    case kAllocVtcm: {
      LOG("device: {kAllocVtcm, %lu, %lx}", message_buffer.len, message_buffer.va);
      assert(message_buffer.len == sizeof(MsgAlloc));
      auto* ma = reinterpret_cast<volatile MsgAlloc*>(message_buffer.va);
      void* p = env->alloc.vtcm_alloc(ma->size, ma->align);
      reinterpret_cast<volatile MsgPointer*>(payload_buffer)->va = va(p);
      setMsg(kNone, sizeof(MsgPointer), va(payload_buffer));
      break;
    }
    case kCopy: {
      LOG("device: {kCopy, %lu, %lx}", message_buffer.len, message_buffer.va);
      assert(message_buffer.len == sizeof(MsgCopy));
      auto* mc = reinterpret_cast<volatile MsgCopy*>(message_buffer.va);
      memcpy(pointer(mc->dst), pointer(mc->src), mc->len);
      setMsg(kNone, 0u, 0u);
      break;
    }
    case kLoad: {
      if (env->dl_handle != nullptr) dlclose(env->dl_handle);
      const char* name = static_cast<const char*>(pointer(message_buffer.va));
      // LOG(stderr, "device: dlopen(%s)", name);
      env->dl_handle = dlopen(name, RTLD_LAZY);
      if (env->dl_handle == nullptr) LOG("dlopen: %s\n", dlerror());
      assert(env->dl_handle != nullptr);
      reinterpret_cast<volatile MsgPointer*>(payload_buffer)->va = va(env->dl_handle);
      setMsg(kNone, sizeof(MsgPointer), va(payload_buffer));
      break;
    }
    case kUnload: {
      assert(env->dl_handle != nullptr);
      assert(message_buffer.len == sizeof(MsgPointer));
      auto* mp = reinterpret_cast<volatile MsgPointer*>(message_buffer.va);
      assert(pointer(mp->va) == env->dl_handle);
      dlclose(env->dl_handle);
      env->dl_handle = nullptr;
      setMsg(kNone, 0u, 0u);
      break;
    }
    case kResolve: {
      LOG("device: {kResolve, %lu, %lx}", message_buffer.len, message_buffer.va);
      assert(env->dl_handle != nullptr);
      dlerror();
      const char* name = static_cast<const char*>(pointer(message_buffer.va));
      void* s = dlsym(env->dl_handle, name);
      reinterpret_cast<volatile MsgPointer*>(payload_buffer)->va = va(s);
      setMsg(kNone, sizeof(MsgPointer), va(payload_buffer));
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
      setMsg(kNone, sizeof(uint32_t), va(payload_buffer));
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
          printMsgCall(*t);
          rv = launcher(t, &pcc);
          LOG("device: execution took %lld pcycles", pcc);
        }
        free(t);
      }
      task_queue.clear();
      *reinterpret_cast<volatile uint32_t*>(payload_buffer) = rv;
      setMsg(kNone, sizeof(uint32_t), va(payload_buffer));
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

static void makePathList(const std::string& arg, std::vector<std::string>* list) {
  size_t p = 0, e = arg.size();
  std::vector<char> tmp;

  while (p < e) {
    tmp.clear();
    bool check_next = true;
    size_t i = p;
    for (; i != e; ++i) {
      char c = arg[i];
      if (check_next) {
        if (c == '\\') {
          check_next = false;
          continue;
        } else if (c == ':') {
          break;
        }
      }
      check_next = true;
      tmp.push_back(c);
    }
    if (!tmp.empty()) list->emplace_back(tmp.begin(), tmp.end());
    p = i + 1;
  }
}

static std::string findInPaths(const std::string& filename, const std::string& paths) {
  std::vector<std::string> path_list;
  makePathList(paths, &path_list);

  for (const auto& p : path_list) {
    std::string pf = p + '/' + filename;
    if (access(pf.c_str(), X_OK) == 0) return std::move(pf);
  }
  // If the search failed, try bare filename. If it cannot be loaded,
  // dlerror will print a meaningful message.
  return filename;
}

// Presence of this function indicates that sim_dev is running.
extern "C" int running_in_sim_dev_17bc90206f6cf5a7();
int running_in_sim_dev_17bc90206f6cf5a7() { return 0; }

int main(int argc, char* argv[]) {
  int opt;
  std::string ld_path;
  while ((opt = getopt(argc, argv, "L:")) != -1) {
    switch (opt) {
      case 'L':
        ld_path += ':' + std::string(optarg);
        break;
      case '?':
        LOG("Usage %s: [-L path1[:path2...]]", argv[0]);
        return 1;
    }
  }

  std::string rt_path = findInPaths("libtvm_runtime.so", ld_path);
  LOG("TVM runtime path: %s", rt_path.c_str());

  Environment env;
  acquire_vector_unit(0);

  const char* builtin[] = {
      "libgcc.so",    "libc.so",     "libc++.so",
      "libc++abi.so", "libc++.so.1", "libc++abi.so.1"  // Alternative names.
  };
  dlinit(sizeof(builtin) / sizeof(builtin[0]), const_cast<char**>(builtin));
  void* rt_handle = dlopen(rt_path.c_str(), RTLD_GLOBAL);
  if (rt_handle == nullptr) {
    LOG("error loading TVM runtime: %s", dlerror());
    return 1;
  }

  // When running TVM runtime on Hexagon there is no longer a device
  // for Hexagon, but standalone ops can still refer to it. All of
  // required DeviceAPI's functionality is adequately implemented
  // via the CPU device, so remap device_api.hexagon to device_api.cpu.
  auto* get_global =
      reinterpret_cast<decltype(&TVMFuncGetGlobal)>(dlsym(rt_handle, "TVMFuncGetGlobal"));
  assert(get_global != nullptr);
  auto* register_global =
      reinterpret_cast<decltype(&TVMFuncRegisterGlobal)>(dlsym(rt_handle, "TVMFuncRegisterGlobal"));
  assert(register_global != nullptr);

  TVMFunctionHandle cpu_api;
  if (get_global("device_api.cpu", &cpu_api) != 0 ||
      register_global("device_api.hexagon", cpu_api, true) != 0) {
    LOG("error setting device_api.hexagon");
    return 1;
  }

  while (!dispatch(&env)) {
  }

  dlclose(rt_handle);
  release_vector_unit();
  return 0;
}
