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
#include "hexagon_buffer_map.h"
#include "hexagon_common.h"
#include "qurt.h"

namespace tvm {
namespace runtime {
namespace hexagon {

#define DBG(msg) LOG(WARNING) << msg << "\n"
#define STR(num) std::to_string(reinterpret_cast<unsigned>(num))
#define HEX(num) "0x" << std::hex << reinterpret_cast<unsigned>(num) << std::dec

//! \brief Minimum stack size in bytes per thread.
#define MIN_STACK_SIZE_BYTES 0x400  // 1KB
//! \brief Maximum stack size in bytes per thread.
#define MAX_STACK_SIZE_BYTES 0x10000  // 64KB
//! \brief Minimum pipe (or command buffer) size in words (or commands) per thread.
#define MIN_PIPE_SIZE_WORDS 10
//! \brief Maximum pipe (or command buffer) size in words (or commands) per thread.
#define MAX_PIPE_SIZE_WORDS 0x10000  // 64K words

class HexagonThreadManager {
  //! \brief Void function.
  typedef void (*voidfunc)(void*);
  //! \brief Semaphore ID.
  typedef unsigned SyncPoint;
  //! \brief Alignment of underlying memory allocations.
  const unsigned MEM_ALIGNMENT = 32;

 public:
  /*!
   * \brief Spawn a number of Hexagon threads with a given stack (in bytes) and pipe (a.k.a. command
   * buffer; in words or commands) within the min and max values specified above. \param num_threads
   * Number of threads to spawn. \param thread_stack_size_bytes Stack size in bytes per thread.
   * \param thread_pipe_size_words Pipe (or command buffer) size in words (or commands).
   */
  HexagonThreadManager(unsigned, unsigned thread_stack_size_bytes, unsigned thread_pipe_size_words);

  //! \brief Destructor
  ~HexagonThreadManager();

  /*!
   * \brief Get the spawned threads as stream handles.
   * \returns Vector of stream handles.
   */
  void GetStreamHandles(std::vector<TVMStreamHandle>* out);

  /*!
   * \brief Non-blocking dispatch of a void function and args on a given thread.
   * \param thread Stream handle of the thread on which to dispatch the void function.
   * \param f Void function to be dispatched.
   * \param args Arguments to pass to the void function.
   * \returns Boolean value indicating success or failure of the dispatch; user must either 1)
   * `Start` threads executing to clear space in the pipe before retrying dispatch or 2) create a
   * `HexagonThreadManager` with a larger pipe.
   */
  bool Dispatch(TVMStreamHandle thread, voidfunc f, void* args);
  /*!
   * \brief Non-blocking signal of a semaphore with a given ID.
   * \param thread Stream handle of the thread which will signal the semaphore.
   * \param syncID ID of the semaphore to be signaled.
   * \returns Boolean value indicating success or failure of the dispatch of the signal; user must
   * either 1) `Start` threads executing to clear space in the pipe before retrying dispatch or 2)
   * create a `HexagonThreadManager` with a larger pipe.
   */
  bool Signal(TVMStreamHandle thread, SyncPoint syncID);
  /*!
   * \brief Non-blocking wait on a semaphore with a given ID.
   * \param thread Stream handle of the thread which will wait on the semaphore.
   * \param syncID ID of the semaphore on which to wait.
   * \returns Boolean value indicating success or failure of the dispatch of the wait; user must
   * either 1) `Start` threads executing to clear space in the pipe before retrying dispatch or 2)
   * create a `HexagonThreadManager` with a larger pipe.
   */
  bool Wait(TVMStreamHandle thread, SyncPoint syncID);
  /*!
   *! \brief Creates a synchronization point between two threads by creating a semaphore,
   *dispatching the `signal_thread` to signal that semaphore and dispatching the `wait_thread to
   *wait on that semaphore. \param signal_thread Stream handle for the thread which will signal the
   *semaphore. \param wait_thread Stream handle for the thread which will wait on the semaphore.
   * \returns Boolean value indicating success or failure of the combined dispatch of both the
   *signal and the wait; user must either 1) `Start` threads executing to clear space in the pipe
   *before retrying dispatch or 2) create a `HexagonThreadManager` with a larger pipe.
   */
  bool SyncFromTo(TVMStreamHandle signal_thread, TVMStreamHandle wait_thread);
  //! \brief Unblock threads to start execution.
  void Start();
  //! \brief Unblock threads to start execution if `Start` has not already been called; blocking
  //! call to wait until all threads have empty pipes.
  void WaitOnThreads();

 private:
  struct ThreadContext {
    HexagonThreadManager* tm;
    unsigned index;
    ThreadContext(HexagonThreadManager* tm, unsigned index) : tm(tm), index(index) {}
  };

  //! \brief Helper function for the constructor to spawn threads.
  void SpawnThreads(unsigned thread_stack_size_bytes, unsigned thread_pipe_size_words);

  //! \brief Helper function for `Signal` and `Wait` to create, initialize and map semaphores by ID.
  void CheckSemaphore(unsigned syncID);

  //! \brief Void function executed by a thread to signal a semaphore.
  static void thread_signal(void* semaphore);

  //! \brief Void function executed by a thread to wait on a semaphore; used by `Wait`.
  static void thread_wait(void* semaphore);

  //! \brief Void function executed by a thread to wait on and free a semaphore; used by
  //! `SyncFromTo`.
  static void thread_wait_free(void* semaphore);

  //! \brief Void function executed by a thread to exit at time of destruction.
  static void thread_exit(void* status);

  //! \brief Void function executed by each thread as `main`.
  static void thread_main(void* context);

  //! \brief Manages underlaying HexagonBuffer allocations.
  HexagonBufferMap hexbuffs;

  //! \brief Number of threads allocatted.
  unsigned nthreads{0};

  //! \brief Pointer to the base of the stacks allocated for all threads; size = `nthreads` *
  //! `thread_stack_size_bytes`.
  void* stack_buffer{nullptr};

  //! \brief Pointer to the base of the pipes (or command buffers) allocated for all threads; size =
  //! `nthreads` * `thread_pipe_size_words` * sizeof(word).
  void* pipe_buffer{nullptr};

  //! \brief QURT thread structure for each spawned thread.
  std::vector<qurt_thread_t> threads;

  //! \brief QURT pipe (or command buffer) structure for each spawned thread.
  std::vector<qurt_pipe_t> pipes;

  //! \brief Thread context passed into each `thread_main` function.
  std::vector<ThreadContext*> contexts;

  //! \brief Semaphores used by `Signal` and `Wait` mapped by ID.
  std::unordered_map<unsigned, qurt_sem_t*> semaphores;

  //! \brief Start semaphore created at time of construction; signled by `Start`.
  qurt_sem_t start_semaphore;

  /*!
   *\brief Encapsulate a void function pointer + arg pointer; sent via pipe to threads to execute.
   */
  struct Command {
    voidfunc f;
    void* args;
    Command(voidfunc f, void* args) : f(f), args(args) {}
  };
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_THREADMANAGER_H_
