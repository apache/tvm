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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_THREAD_MANAGER_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_THREAD_MANAGER_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hexagon_buffer.h"
#include "hexagon_buffer_manager.h"
#include "hexagon_common.h"
#include "hexagon_htp.h"
#include "hexagon_hvx.h"
#include "qurt.h"

namespace tvm {
namespace runtime {
namespace hexagon {

typedef enum {
  NONE = -1,
  DMA_0 = 0,
  HTP_0,
  HVX_0,
  HVX_1,
  HVX_2,
  HVX_3,
  MAX,
} HardwareResourceType;

class HexagonThreadManager {
  //! \brief Void function.
  using voidfunc = void (*)(void*);
  //! \brief Semaphore ID.
  using SyncPoint = unsigned;
  //! \brief Alignment of underlying memory allocations.
  const unsigned MEM_ALIGNMENT = 32;
  //! \brief Minimum stack size in bytes per thread.
  const unsigned MIN_STACK_SIZE_BYTES = 0x400;  // 1KB
  //! \brief Maximum stack size in bytes per thread.
  const unsigned MAX_STACK_SIZE_BYTES = 0x10000;  // 64KB
  //! \brief Minimum pipe (or command buffer) size in words (or commands) per thread.
  const unsigned MIN_PIPE_SIZE_WORDS = 10;
  //! \brief Maximum pipe (or command buffer) size in words (or commands) per thread.
  const unsigned MAX_PIPE_SIZE_WORDS = 0x10000;  // 64K words

 public:
  /*!
   * \brief Spawn a number of Hexagon threads with a given stack (in bytes) and pipe (a.k.a. command
   * buffer; in words or commands) within the min and max values specified above.
   * \param num_threads Number of threads to spawn.
   * \param thread_stack_size_bytes Stack size in bytes per thread.
   * \param thread_pipe_size_words Pipe (or command buffer) size in words (or commands).
   */
  HexagonThreadManager(unsigned, unsigned thread_stack_size_bytes, unsigned thread_pipe_size_words,
                       const std::vector<HardwareResourceType> = {});

  //! \brief Destructor
  ~HexagonThreadManager();

  /*!
   * \brief Get the spawned threads as stream handles.
   * \returns Vector of stream handles.
   */
  const std::vector<TVMStreamHandle> GetStreamHandles();

  /*!
   * \brief Get the spawned threads as stream handles for a resource type.
   * \returns stream handle.
   */
  TVMStreamHandle GetStreamHandleByResourceType(HardwareResourceType type);

  /*!
   * \brief Get the resource type for a stream handle
   * \returns stream handle.
   */
  HardwareResourceType GetResourceTypeForStreamHandle(TVMStreamHandle thread);

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
   * \brief Creates a synchronization point between two threads by creating a semaphore,
   *dispatching the `signal_thread` to signal that semaphore and dispatching the `wait_thread to
   *wait on that semaphore.
   * \param signal_thread Stream handle for the thread which will signal the
   *semaphore.
   * \param wait_thread Stream handle for the thread which will wait on the semaphore.
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
    qurt_pipe_t* pipe;
    unsigned index;
    HardwareResourceType resource_type;
    HexagonHvx* hvx;
    HexagonHtp* htp;
    uint64_t status;
    ThreadContext(qurt_pipe_t* pipe, unsigned index, HardwareResourceType resource_type,
                  HexagonHvx* hvx, HexagonHtp* htp)
        : pipe(pipe), index(index), resource_type(resource_type), hvx(hvx), htp(htp), status(0) {
      CHECK(resource_type == NONE || (hvx && htp))
          << "Missing resource manager pointer, type: " << resource_type << " hvx: " << hvx
          << " htp: " << htp;
    }
  };

  //! \brief Helper function to ensure the set of requested resources is valid.
  void CheckResources();

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
  static void thread_exit(void* context);

  //! \brief Void function executed by each thread as `main`.
  static void thread_main(void* context);

  //! \brief Manages underlying HexagonBuffer allocations.
  HexagonBufferManager hexbuffs_;

  //! \brief Number of threads allocatted.
  unsigned nthreads_{0};

  //! \brief Pointer to the base of the stacks allocated for all threads; size = `nthreads` *
  //! `thread_stack_size_bytes`.
  void* stack_buffer_{nullptr};

  //! \brief Pointer to the base of the pipes (or command buffers) allocated for all threads; size =
  //! `nthreads` * `thread_pipe_size_words` * sizeof(word).
  void* pipe_buffer_{nullptr};

  //! \brief QURT thread structure for each spawned thread.
  std::vector<qurt_thread_t> threads_;

  //! \brief QURT pipe (or command buffer) structure for each spawned thread.
  std::vector<qurt_pipe_t> pipes_;

  //! \brief Thread context passed into each `thread_main` function.
  std::vector<ThreadContext*> contexts_;

  //! \brief Semaphores used by `Signal` and `Wait` mapped by ID.
  std::unordered_map<unsigned, qurt_sem_t*> semaphores_;

  //! \brief Protects updates to semaphores_
  std::mutex semaphores_mutex_;

  //! \brief Start semaphore created at time of construction; signled by `Start`.
  qurt_sem_t start_semaphore_;

  /*!
   *\brief Encapsulate a void function pointer + arg pointer; sent via pipe to threads to execute.
   */
  struct Command {
    voidfunc f;
    void* args;
    Command(voidfunc f, void* args) : f(f), args(args) {}
  };

  //! \brief List of hardware resources
  std::vector<HardwareResourceType> hw_resources_;

  //! \brief Whether or not resource managers should be created
  bool create_resource_managers_{false};

  //! \brief HTP hardware resource.
  // TODO(HWE): Move binding of HTP to a specific thread
  std::unique_ptr<HexagonHtp> htp_;

  //! \brief HVX hardware resource.
  // TODO(HWE): Move binding of individual HVX instances to a specific thread
  std::unique_ptr<HexagonHvx> hvx_;
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_THREAD_MANAGER_H_
