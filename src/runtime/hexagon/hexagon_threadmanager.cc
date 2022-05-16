#include "hexagon_threadmanager.h"

#if defined(__hexagon__)

#include <stdlib.h>

namespace tvm {
namespace runtime {
namespace hexagon {

HexagonThreadManager::HexagonThreadManager(unsigned num_threads, unsigned thread_stack_size_bytes, unsigned thread_pipe_size_words) {
  CHECK_LE(num_threads, QURT_MAX_HTHREAD_LIMIT);
  this->nthreads = num_threads;
  
  number_semaphores = 0;
  semaphores = NULL;
  threads = NULL;
  pipes = NULL;
  contexts = NULL;

  this->thread_stack_size = thread_stack_size_bytes;
  this->thread_pipe_size_words = thread_pipe_size_words;
  this->thread_pipe_size = thread_pipe_size_words * sizeof(qurt_pipe_data_t);
  
  // Allocate all stack space for threads
  int ret = posix_memalign(&stack_buffer, MEM_ALIGNMENT, thread_stack_size_bytes * nthreads);
  CHECK_EQ(ret, 0);
  
  // Allocate space for pipe buffers (command queues)
  ret = posix_memalign(&pipe_buffer, MEM_ALIGNMENT, thread_pipe_size * nthreads);
  CHECK_EQ(ret, 0);

  SpawnThreads();
}

HexagonThreadManager::~HexagonThreadManager() {
  
  // dispatch a command to each thread to exit with status 0
  for (int i = 0; i < nthreads; i++) {
    Dispatch(i, &thread_exit, (void*) 0);
  }

  // join with each thread (wait for them to terminate); if already exited, call returns immediately
  int status;  // don't actually care what the thread exit status was
  for (int i = 0; i < nthreads; i++) {
    qurt_thread_join(threads[i], &status);
  }

  // Delete pipe objects and contexts
  for (int i = 0; i < nthreads; i++) {
    if (pipes != NULL) {
      qurt_pipe_delete(&pipes[i]);  // with manual buffers, cannot use qurt_pipe_destroy()
    }
    if (contexts != NULL) {
      delete contexts[i];
    }
  }

  // Dealloc semaphores
  if (number_semaphores > 0) {
    for (int i = 0; i < number_semaphores; i++) {
      qurt_sem_destroy(&semaphores[i]);
    }
    free(semaphores);
  }

  // Dealloc memory blocks
  free(threads);
  free(pipes);
  free(contexts);
  free(stack_buffer);
  free(pipe_buffer);
}

void HexagonThreadManager::SpawnThreads() {
  char* next_stack_start = (char*) stack_buffer;
  char* next_pipe_start = (char*) pipe_buffer;

  int ret;
  // array of thread objects
  posix_memalign((void**)&threads, MEM_ALIGNMENT, sizeof(qurt_thread_t) * nthreads);
  CHECK_EQ(ret, 0);
  // array of pipe objects
  ret = posix_memalign((void**)&pipes, MEM_ALIGNMENT, sizeof(qurt_pipe_t) * nthreads);
  CHECK_EQ(ret, 0);
  // array of ThreadContexts
  ret = posix_memalign((void**)&contexts, MEM_ALIGNMENT, sizeof(ThreadContext) * nthreads);
  CHECK_EQ(ret, 0);
  
  // First, create pipe resources for all threads
  for (int i = 0; i < nthreads; i++) {
    qurt_pipe_attr_t pipe_attr;
    qurt_pipe_attr_init(&pipe_attr);
    qurt_pipe_attr_set_buffer(&pipe_attr, (qurt_pipe_data_t*) next_pipe_start);
    next_pipe_start += thread_pipe_size;
    qurt_pipe_attr_set_buffer_partition(&pipe_attr, QURT_PIPE_ATTR_MEM_PARTITION_RAM);
    qurt_pipe_attr_set_elements(&pipe_attr, thread_pipe_size_words);

    // create the pipe
    int rc = qurt_pipe_init(&pipes[i], &pipe_attr);
    CHECK_EQ(rc, QURT_EOK);
  }

  // Create all threads
  for (int i = 0; i < nthreads; i++) {
    // create initialize the thread attr
    qurt_thread_attr_t thread_attr;
    qurt_thread_attr_init(&thread_attr);
    qurt_thread_attr_set_stack_addr(&thread_attr, next_stack_start);
    next_stack_start += thread_stack_size;
    qurt_thread_attr_set_stack_size(&thread_attr, thread_stack_size);

    // create the thread
    contexts[i] = new ThreadContext(this, i);
    int rc = qurt_thread_create(&threads[i], &thread_attr, thread_main, (void*)&contexts[i]);
    CHECK_EQ(rc, QURT_EOK);
  }
}

void HexagonThreadManager::GetStreamHandles(std::vector<TVMStreamHandle>* out) {
  for (int i = 0; i < nthreads; i++) {
    out->push_back(static_cast<TVMStreamHandle>((void*)i));  // threads identified by index into `threads` array
  }
}
  
/*
AllocateSyncs is not necessary if a program ONLY uses dynamically-allocated semaphores --- i.e., `SyncFromTo` is the only
sync mechanism called.
*/
void HexagonThreadManager::AllocateSyncs(unsigned number_syncs) {
  number_semaphores = number_syncs;
  semaphores = (qurt_sem_t*) malloc(sizeof(qurt_sem_t) * number_syncs);
  for (int i = 0; i < number_syncs; i++) {
    qurt_sem_init_val(&semaphores[i], 0);
  }
}

void HexagonThreadManager::Dispatch(unsigned thread, voidfunc f, void* args) {
  Command* cmd = new Command(f, args);
  qurt_pipe_data_t msg = (qurt_pipe_data_t)(cmd);
  qurt_pipe_t* pipeAddr = &pipes[thread];

  int trysend = qurt_pipe_try_send(pipeAddr, msg);
  if (trysend) {
    // log that the pipe was full, then do a blocking send
    DLOG(INFO) << "Blocking on dispatch to thread " << thread << " due to full pipe\n";
    qurt_pipe_send(pipeAddr, msg);
  }
}

void HexagonThreadManager::Signal(unsigned thread, unsigned syncID) {
  CHECK_LT(syncID, number_semaphores);
  qurt_sem_t* semaphore = &semaphores[syncID];
  Dispatch(thread, thread_signal, (void*) semaphore);
}

void HexagonThreadManager::Wait(unsigned thread, unsigned syncID) {
  CHECK_LT(syncID, number_semaphores);
  qurt_sem_t* semaphore = &semaphores[syncID];
  Dispatch(thread, thread_wait, (void*) semaphore);
}

/* Create a sync_from_to relationship with a dynamic semaphore allocation.
Makes use of thread_wait_free to also free the semaphore after sync is complete.
*/
void HexagonThreadManager::SyncFromTo(unsigned signal_thread, unsigned wait_thread) {
  qurt_sem_t* sem = (qurt_sem_t*) malloc(sizeof(qurt_sem_t));
  qurt_sem_init_val(sem, 0);
  Dispatch(signal_thread, thread_signal, (void*)sem);
  Dispatch(wait_thread, thread_wait_free, (void*)sem);
}
  
 void HexagonThreadManager::thread_signal(void* semaphore) {
  qurt_sem_add( (qurt_sem_t*) semaphore, QURT_MAX_HTHREAD_LIMIT);
}

 void HexagonThreadManager::thread_wait(void* semaphore) {
  qurt_sem_down( (qurt_sem_t*) semaphore);
}

/* Wait on the passed semaphore object, then free it. */
 void HexagonThreadManager::thread_wait_free(void* semaphore) {
  qurt_sem_down((qurt_sem_t*)semaphore);  // blocks until signal is complete
  qurt_sem_destroy((qurt_sem_t*)semaphore);
  free(semaphore);
}

 void HexagonThreadManager::thread_exit(void* status) {
  qurt_thread_exit( (int) status);
}

 void HexagonThreadManager::thread_main(void* context) {
  ThreadContext* tc = static_cast<ThreadContext*>(context);
  unsigned index = tc->index;
  qurt_pipe_t* mypipe = &(tc->tm->pipes[index]);
   
  while (true) {  // loop, executing commands from pipe
    qurt_pipe_data_t msg = qurt_pipe_receive(mypipe);  // blocks if empty
    Command* cmd = (Command*)(msg);
    voidfunc f = cmd->f;
    void* args = cmd->args;
    delete cmd;  // reclaim memory before call in case call is thread_exit
    f(args);
  }
  // thread exit is handled by dispatching an exit command
}


}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // __hexagon__
