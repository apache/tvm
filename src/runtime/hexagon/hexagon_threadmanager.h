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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_THREADMANAGER_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_THREADMANAGER_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hexagon_buffer.h"
#include "hexagon_common.h"
#include "qurt.h"

namespace tvm {
namespace runtime {
namespace hexagon {

// TODO(Straw): move to separate file and use inside Hexagon Device API
class HexagonBufferMap {
 public:
  void FreeHexagonBuffer(void* ptr) {
    auto it = hexagon_buffer_map_.find(ptr);
    CHECK(it != hexagon_buffer_map_.end())
        << "Attempt made to free unknown or already freed dataspace allocation";
    CHECK(it->second != nullptr);
    hexagon_buffer_map_.erase(it);
  }
  template <typename... Args>
  void* AllocateHexagonBuffer(Args&&... args) {
    auto buf = std::make_unique<HexagonBuffer>(std::forward<Args>(args)...);
    void* ptr = buf->GetPointer();
    hexagon_buffer_map_.insert({ptr, std::move(buf)});
    return ptr;
  }

 private:
  std::unordered_map<void*, std::unique_ptr<HexagonBuffer>> hexagon_buffer_map_;
};

#define DBG(msg) LOG(WARNING) << msg << "\n"
#define STR(num) std::to_string(reinterpret_cast<unsigned>(num))
#define HEX(num) "0x" << std::hex << reinterpret_cast<unsigned>(num) << std::dec

#define MIN_STACK_SIZE_BYTES 0x400    // 1KB
#define MAX_STACK_SIZE_BYTES 0x10000  // 64KB
#define MIN_PIPE_SIZE_WORDS 10
#define MAX_PIPE_SIZE_WORDS 0x10000  // 64K words

class HexagonThreadManager {
  typedef void (*voidfunc)(void*);
  typedef unsigned SyncPoint;
  const unsigned MEM_ALIGNMENT = 32;

 public:
  HexagonThreadManager(unsigned num_threads, unsigned thread_stack_size_bytes,
                       unsigned thread_pipe_size_words);
  ~HexagonThreadManager();
  void GetStreamHandles(std::vector<TVMStreamHandle>* out);
  bool Dispatch(TVMStreamHandle thread, voidfunc f, void* args);
  bool Dispatch(TVMStreamHandle thread, PackedFunc f, TVMArgs args, TVMRetValue* rv = NULL);
  bool Signal(TVMStreamHandle thread, SyncPoint syncID);
  bool Wait(TVMStreamHandle thread, SyncPoint syncID);
  bool SyncFromTo(TVMStreamHandle signal_thread, TVMStreamHandle wait_thread);
  void Start();          // Unblock threads to start execution
  void WaitOnThreads();  // Blocking call to wait until all threads have empty queues

 private:
  struct ThreadContext {
    HexagonThreadManager* tm;
    unsigned index;
    ThreadContext(HexagonThreadManager* tm, unsigned index) : tm(tm), index(index) {}
  };

  void SpawnThreads(unsigned thread_stack_size_bytes, unsigned thread_pipe_size_words);
  void CheckSemaphore(unsigned syncID);
  static void thread_signal(void* semaphore);
  static void thread_wait(void* semaphore);
  static void thread_wait_free(void* semaphore);
  static void thread_exit(void* status);
  static void thread_unpack(void* wrapped);
  static void thread_main(void* context);

  HexagonBufferMap hexbuffs;
  unsigned nthreads{0};
  void* stack_buffer{nullptr};
  void* pipe_buffer{nullptr};
  std::vector<qurt_thread_t> threads;
  std::vector<qurt_pipe_t> pipes;
  std::vector<ThreadContext*> contexts;
  std::unordered_map<unsigned, qurt_sem_t*> semaphores;
  qurt_sem_t start_semaphore;

  /*
    Encapsulate a function pointer + arg pointer. Sent via pipe to threads to execute.
    Function should have type   "void myfunc(void* args)"
  */
  struct Command {
    voidfunc f;
    void* args;
    Command(voidfunc f, void* args) : f(f), args(args) {}
  };

  /*
    Encapsulate a PackedFunc + args + return value pointer.
    Used to wrap a PackedFunc call into a single object for use in a Command.
   */
  struct WrappedPackedFunc {
    PackedFunc f;
    TVMArgs args;
    TVMRetValue* rv;
    WrappedPackedFunc(PackedFunc f, TVMArgs args, TVMRetValue* rv) : f(f), args(args), rv(rv) {}
  };
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_THREADMANAGER_H_
