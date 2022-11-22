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

#include "hexagon_thread_manager.h"

namespace tvm {
namespace runtime {
namespace hexagon {

HexagonThreadManager::HexagonThreadManager(unsigned num_threads, unsigned thread_stack_size_bytes,
                                           unsigned thread_pipe_size_words,
                                           const std::vector<HardwareResourceType> hw_resources) {
  // Note: could technically manage more software threads than allowable hardware threads, but
  // there is no system constant defined in the qurt libs for that maximum.
  CHECK(num_threads);
  CHECK_LE(num_threads, QURT_MAX_HTHREAD_LIMIT);
  nthreads_ = num_threads;

  CHECK_GE(thread_stack_size_bytes, MIN_STACK_SIZE_BYTES);
  CHECK_LE(thread_stack_size_bytes, MAX_STACK_SIZE_BYTES);

  CHECK_GE(thread_pipe_size_words, MIN_PIPE_SIZE_WORDS);
  CHECK_LE(thread_pipe_size_words, MAX_PIPE_SIZE_WORDS);

  hw_resources_ = hw_resources;
  CheckResources();

  if (create_resource_managers_) {
    DLOG(INFO) << "Initialize hardware resource managers";
    // This creates the manager objects, which reserves (acquires) the resources.
    // Calls to lock/unlock will be performed on threads dedicated to instances.
    // This must be done before spawning threads so we can pass pointers to the
    // objects in the thread context.
    htp_ = std::make_unique<HexagonHtp>();
    hvx_ = std::make_unique<HexagonHvx>();
  }

  DLOG(INFO) << "Spawning threads";
  SpawnThreads(thread_stack_size_bytes, thread_pipe_size_words);

  // Initially, block all threads until we get the Start() call
  qurt_sem_init_val(&start_semaphore_, 0);
  for (unsigned i = 0; i < nthreads_; i++) {
    Dispatch(reinterpret_cast<TVMStreamHandle>(i), thread_wait, &start_semaphore_);
  }
}

HexagonThreadManager::~HexagonThreadManager() {
  // In case Start() was never explicitly called, call it now to prevent deadlock
  if (qurt_sem_get_val(&start_semaphore_) == 0) {
    Start();
  }

  DLOG(INFO) << "Threads started";

  // dispatch a command to each thread to exit with status 0
  for (unsigned i = 0; i < nthreads_; i++) {
    bool success = Dispatch(reinterpret_cast<TVMStreamHandle>(i), thread_exit, contexts_[i]);
    while (!success) {
      success = Dispatch(reinterpret_cast<TVMStreamHandle>(i), thread_exit, contexts_[i]);
    }
  }

  DLOG(INFO) << "Threads exited";

  // join with each thread (wait for them to terminate); if already exited, the call returns
  // immediately
  int status;  // don't actually care what the thread exit status was
  for (unsigned i = 0; i < nthreads_; i++) {
    qurt_thread_join(threads_[i], &status);
  }

  DLOG(INFO) << "Threads joined";

  // Destroy semaphores
  qurt_sem_destroy(&start_semaphore_);
  for (auto it : semaphores_) {
    qurt_sem_destroy(it.second);
    free(it.second);
  }

  DLOG(INFO) << "Semaphores destroyed";

  // Delete pipe objects and contexts
  for (unsigned i = 0; i < nthreads_; i++) {
    qurt_pipe_destroy(&pipes_[i]);
    delete contexts_[i];
  }

  DLOG(INFO) << "Pipes and contexts deleted";

  // Dealloc memory blocks
  hexbuffs_.FreeHexagonBuffer(stack_buffer_);
  hexbuffs_.FreeHexagonBuffer(pipe_buffer_);

  DLOG(INFO) << "Buffers freed";

  // Release hardware
  htp_.reset();
  hvx_.reset();

  DLOG(INFO) << "Hardware resources released";
}

void HexagonThreadManager::CheckResources() {
  create_resource_managers_ = false;
  CHECK(hw_resources_.empty() || hw_resources_.size() == nthreads_)
      << "Thread count must match resource count";
  if (!hw_resources_.empty()) {
    // Ensure that no more than one of each hardware resource is specified
    for (int i = 0; i < hw_resources_.size(); i++) {
      if (hw_resources_[i] != NONE) {
        create_resource_managers_ = true;
        for (int j = i + 1; j < hw_resources_.size(); j++) {
          CHECK(hw_resources_[i] != hw_resources_[j])
              << "No more than one of each resource type may be specified " << hw_resources_[i];
        }
      }
    }
  }
}

void HexagonThreadManager::SpawnThreads(unsigned thread_stack_size_bytes,
                                        unsigned thread_pipe_size_words) {
  // allocate all stack space for threads
  stack_buffer_ = hexbuffs_.AllocateHexagonBuffer(thread_stack_size_bytes * nthreads_,
                                                  MEM_ALIGNMENT, String("global"));
  // allocate space for pipe buffers (command queues)
  unsigned thread_pipe_size_bytes = thread_pipe_size_words * sizeof(qurt_pipe_data_t);
  pipe_buffer_ = hexbuffs_.AllocateHexagonBuffer(thread_pipe_size_bytes * nthreads_, MEM_ALIGNMENT,
                                                 String("global"));

  threads_.resize(nthreads_);
  pipes_.resize(nthreads_);
  contexts_.resize(nthreads_);

  DLOG(INFO) << "Buffers allocated";

  // First, create pipe resources for all threads
  char* next_pipe_start = reinterpret_cast<char*>(pipe_buffer_);
  for (unsigned i = 0; i < nthreads_; i++) {
    qurt_pipe_attr_t pipe_attr;
    qurt_pipe_attr_init(&pipe_attr);
    qurt_pipe_attr_set_buffer(&pipe_attr, reinterpret_cast<qurt_pipe_data_t*>(next_pipe_start));
    next_pipe_start += thread_pipe_size_bytes;
    qurt_pipe_attr_set_buffer_partition(&pipe_attr, QURT_PIPE_ATTR_MEM_PARTITION_RAM);
    qurt_pipe_attr_set_elements(&pipe_attr, thread_pipe_size_words);

    // create the pipe
    int rc = qurt_pipe_init(&pipes_[i], &pipe_attr);
    CHECK_EQ(rc, QURT_EOK);
  }

  DLOG(INFO) << "Pipes created";

  // Create all threads
  char* next_stack_start = reinterpret_cast<char*>(stack_buffer_);
  for (unsigned i = 0; i < nthreads_; i++) {
    // create initialize the thread attr
    qurt_thread_attr_t thread_attr;
    char name[32];
    qurt_thread_attr_init(&thread_attr);
    qurt_thread_attr_set_stack_addr(&thread_attr, next_stack_start);
    qurt_thread_attr_set_stack_size(&thread_attr, thread_stack_size_bytes);
    snprintf(name, sizeof(name), "thread %d", i);
    qurt_thread_attr_set_name(&thread_attr, name);
    next_stack_start += thread_stack_size_bytes;

    // create the thread
    contexts_[i] = new ThreadContext(&pipes_[i], i, hw_resources_.empty() ? NONE : hw_resources_[i],
                                     hvx_.get(), htp_.get());
    int rc = qurt_thread_create(&threads_[i], &thread_attr, thread_main, contexts_[i]);
    CHECK_EQ(rc, QURT_EOK);
  }

  DLOG(INFO) << "Threads created";
}

const std::vector<TVMStreamHandle> HexagonThreadManager::GetStreamHandles() {
  std::vector<TVMStreamHandle> out;
  for (unsigned i = 0; i < nthreads_; i++) {
    // threads identified by index into `threads` array
    out.push_back(reinterpret_cast<TVMStreamHandle>(i));
  }
  return out;
}

TVMStreamHandle HexagonThreadManager::GetStreamHandleByResourceType(HardwareResourceType type) {
  for (unsigned i = 0; i < hw_resources_.size(); i++) {
    if (hw_resources_[i] == type) {
      return reinterpret_cast<TVMStreamHandle>(i);
    }
  }
  CHECK(false) << "Thread for resource type " << type << " not found";
}

HardwareResourceType HexagonThreadManager::GetResourceTypeForStreamHandle(TVMStreamHandle thread) {
  CHECK(hw_resources_.size() > reinterpret_cast<int>(thread))
      << "No thread for handle id exists " << thread;
  return hw_resources_[reinterpret_cast<int>(thread)];
}

bool HexagonThreadManager::Dispatch(TVMStreamHandle stream, voidfunc f, void* args) {
  unsigned thread = reinterpret_cast<unsigned>(stream);
  DLOG(INFO) << "Dispatching to stream " << thread;
  Command* cmd = new Command(f, args);  // Command object freed by receiving thread
  qurt_pipe_data_t msg = (qurt_pipe_data_t)(cmd);
  qurt_pipe_t* pipeAddr = &pipes_[thread];

  int trysend = qurt_pipe_try_send(pipeAddr, msg);
  return trysend == 0;
}

void HexagonThreadManager::Start() { thread_signal(&start_semaphore_); }

void HexagonThreadManager::WaitOnThreads() {
  // Using standard signal mechanism to block the "main" thread on all worker threads.
  // Note: this would be slightly more efficient as a barrier, but would need some extra code to
  //  wait on the barrier that would only be used once.

  // In case Start() was never explicitly called, call it now to prevent deadlock
  if (qurt_sem_get_val(&start_semaphore_) == 0) {
    Start();
  }

  std::vector<qurt_sem_t> finished;
  finished.resize(nthreads_);

  // initialize one semaphore for each thread
  for (unsigned i = 0; i < nthreads_; i++) {
    qurt_sem_init_val(&finished[i], 0);
  }
  // dispatch signal() command to each thread on their private semaphore
  for (unsigned i = 0; i < nthreads_; i++) {
    bool success = Dispatch(reinterpret_cast<TVMStreamHandle>(i), thread_signal, &finished[i]);
    while (!success) {
      success = Dispatch(reinterpret_cast<TVMStreamHandle>(i), thread_signal, &finished[i]);
    }
  }
  // wait on each semaphore, one at a time
  for (unsigned i = 0; i < nthreads_; i++) {
    thread_wait(&finished[i]);
  }

  // clean up
  for (unsigned i = 0; i < nthreads_; i++) {
    qurt_sem_destroy(&finished[i]);
  }
}

void HexagonThreadManager::CheckSemaphore(unsigned syncID) {
  if (semaphores_.find(syncID) == semaphores_.end()) {
    semaphores_[syncID] = reinterpret_cast<qurt_sem_t*>(malloc(sizeof(qurt_sem_t)));
    qurt_sem_init_val(semaphores_[syncID], 0);
  }
}

bool HexagonThreadManager::Signal(TVMStreamHandle thread, SyncPoint syncID) {
  CheckSemaphore(syncID);
  DLOG(INFO) << "Dispatching signal to thread " << thread << " on semaphore ID " << syncID
             << " located @ 0x" << std::hex << semaphores_[syncID];
  return Dispatch(thread, thread_signal, semaphores_[syncID]);
}

bool HexagonThreadManager::Wait(TVMStreamHandle thread, SyncPoint syncID) {
  CheckSemaphore(syncID);
  DLOG(INFO) << "Dispatching wait to thread " << thread << " on semaphore ID " << syncID
             << " located @ 0x" << std::hex << semaphores_[syncID];
  return Dispatch(thread, thread_wait, semaphores_[syncID]);
}

/* Create a sync_from_to relationship with a dynamic semaphore allocation.
Makes use of thread_wait_free to also free the semaphore after sync is complete.
*/
bool HexagonThreadManager::SyncFromTo(TVMStreamHandle signal_thread, TVMStreamHandle wait_thread) {
  qurt_sem_t* sem = reinterpret_cast<qurt_sem_t*>(malloc(sizeof(qurt_sem_t)));
  qurt_sem_init_val(sem, 0);
  if (Dispatch(signal_thread, thread_signal, sem)) {
    return Dispatch(wait_thread, thread_wait_free, sem);
  } else {
    return false;
  }
}

void HexagonThreadManager::thread_signal(void* semaphore) {
  DLOG(INFO) << "Signaling semaphore addr 0x" << std::hex << semaphore;
  qurt_sem_add(reinterpret_cast<qurt_sem_t*>(semaphore), QURT_MAX_HTHREAD_LIMIT);
}

void HexagonThreadManager::thread_wait(void* semaphore) {
  DLOG(INFO) << "Waiting on semaphore addr 0x" << std::hex << semaphore;
  qurt_sem_down(reinterpret_cast<qurt_sem_t*>(semaphore));
}

/* Wait on the passed semaphore object, then free it. */
void HexagonThreadManager::thread_wait_free(void* semaphore) {
  qurt_sem_down(reinterpret_cast<qurt_sem_t*>(semaphore));  // blocks until signal is complete
  qurt_sem_destroy(reinterpret_cast<qurt_sem_t*>(semaphore));
  free(semaphore);
}

void HexagonThreadManager::thread_exit(void* context) {
  ThreadContext* tc = static_cast<ThreadContext*>(context);
  unsigned index = tc->index;
  HardwareResourceType resource_type = tc->resource_type;

  if ((resource_type == HVX_0) || (resource_type == HVX_1) || (resource_type == HVX_2) ||
      (resource_type == HVX_3)) {
    tc->hvx->Unlock();
    DLOG(INFO) << "Thread " << index << " unlocked an HVX instance";
  } else if (resource_type == HTP_0) {
    // TODO(HWE): Perform HTP lock/unlock in thread instead of HexagonHtp
    // tc->htp->Unlock();
    // DLOG(INFO) << "Thread " << index << " unlocked the HTP";
  }

  DLOG(INFO) << "Thread " << index << " exiting";
  qurt_thread_exit((uint64_t)tc->status);
}

void HexagonThreadManager::thread_main(void* context) {
  ThreadContext* tc = static_cast<ThreadContext*>(context);
  unsigned index = tc->index;
  qurt_pipe_t* mypipe = tc->pipe;
  HardwareResourceType resource_type = tc->resource_type;

  DLOG(INFO) << "Thread " << index << " spawned";

  if ((resource_type == HVX_0) || (resource_type == HVX_1) || (resource_type == HVX_2) ||
      (resource_type == HVX_3)) {
    tc->hvx->Lock();
    DLOG(INFO) << "Thread " << index << " locked an HVX instance";
  } else if (resource_type == HTP_0) {
    // TODO(HWE): Perform HTP lock/unlock in thread instead of HexagonHtp
    // tc->htp->Lock();
    // DLOG(INFO) << "Thread " << index << " locked the HTP";
  }

  while (true) {  // loop, executing commands from pipe
    DLOG(INFO) << "Thread " << index << " receiving command";
    qurt_pipe_data_t msg = qurt_pipe_receive(mypipe);  // blocks if empty
    Command* cmd = reinterpret_cast<Command*>(msg);
    voidfunc f = cmd->f;
    void* args = cmd->args;
    delete cmd;
    f(args);
  }
  // thread exit is handled by dispatching an exit command
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
