#include "hexagon_threadmanager.h"

#if defined(__hexagon__)

#include <stdlib.h>
#include <sstream>

namespace tvm {
namespace runtime {
namespace hexagon {
  
HexagonThreadManager::HexagonThreadManager(unsigned num_threads, unsigned thread_stack_size_bytes, unsigned thread_pipe_size_words) {
  // Note: could technically manage more software threads than allowable hardware threads, but there is no system constant defined
  //  in the qurt libs for that maximum.
  CHECK(num_threads);
  CHECK_LE(num_threads, QURT_MAX_HTHREAD_LIMIT);
  nthreads = num_threads;

  CHECK_LE(thread_stack_size_bytes, MAX_STACK_SIZE_BYTES);
  if (thread_stack_size_bytes < MIN_STACK_SIZE_BYTES) {
    thread_stack_size_bytes = MIN_STACK_SIZE_BYTES;
  }

  CHECK_LE(thread_pipe_size_words, MAX_PIPE_SIZE_WORDS);
  if (thread_pipe_size_words < MIN_PIPE_SIZE_WORDS) {
    thread_pipe_size_words = MIN_PIPE_SIZE_WORDS;
  }

  DBG("Spawning threads");
  SpawnThreads(thread_stack_size_bytes, thread_pipe_size_words);

  // Initially, block all threads until we get the Start() call
  qurt_sem_init_val(&start_semaphore, 0);
  for (int i = 0; i < nthreads; i ++) {
    Dispatch((TVMStreamHandle)i, thread_wait, &start_semaphore);
  }
}

HexagonThreadManager::~HexagonThreadManager() {

  // In case Start() was never explicitly called, call it now to prevent deadlock
  if (qurt_sem_get_val(&start_semaphore) == 0) {
    Start();
  }
  
  // dispatch a command to each thread to exit with status 0
  for (int i = 0; i < nthreads; i++) {
    while(!Dispatch((TVMStreamHandle)i, &thread_exit, (void*) 0));
  }

  // join with each thread (wait for them to terminate); if already exited, the call returns immediately
  int status;  // don't actually care what the thread exit status was
  for (int i = 0; i < nthreads; i++) {
    qurt_thread_join(threads[i], &status);
  }

  // destroy semaphores
  qurt_sem_destroy(&start_semaphore);
  for (int i = 0; i < semaphores.size(); i++) {
    qurt_sem_destroy(&semaphores[i]);
  }
  
  // Delete pipe objects and contexts
  for (int i = 0; i < nthreads; i++) {
    if (pipes != NULL) {
      qurt_pipe_delete(&pipes[i]);  // with manual buffers, cannot use qurt_pipe_destroy()
    }
    if (contexts != NULL) {
      delete contexts[i]; // TODO: is there a qurt function for this?
    }
  }

  // Dealloc memory blocks
  hexbuffs.FreeHexagonBuffer(threads);
  hexbuffs.FreeHexagonBuffer(pipes);
  hexbuffs.FreeHexagonBuffer(contexts);
  hexbuffs.FreeHexagonBuffer(stack_buffer);
  hexbuffs.FreeHexagonBuffer(pipe_buffer);
}

void HexagonThreadManager::SpawnThreads(unsigned thread_stack_size_bytes, unsigned thread_pipe_size_words) {
  
  // allocate all stack space for threads
  stack_buffer = hexbuffs.AllocateHexagonBuffer(thread_stack_size_bytes * nthreads, MEM_ALIGNMENT, String("global"));
  // allocate space for pipe buffers (command queues)
  unsigned thread_pipe_size_bytes = thread_pipe_size_words * sizeof(qurt_pipe_data_t);
  pipe_buffer = hexbuffs.AllocateHexagonBuffer(thread_pipe_size_bytes * nthreads, MEM_ALIGNMENT, String("global"));
  // array of thread objects
  threads = static_cast<qurt_thread_t*>(hexbuffs.AllocateHexagonBuffer(sizeof(qurt_thread_t) * nthreads, MEM_ALIGNMENT, String("global")));
  // array of pipe objects
  pipes = static_cast<qurt_pipe_t*>(hexbuffs.AllocateHexagonBuffer(sizeof(qurt_pipe_t) * nthreads, MEM_ALIGNMENT, String("global")));
  // array of ThreadContexts
  contexts = static_cast<ThreadContext**>(hexbuffs.AllocateHexagonBuffer(sizeof(ThreadContext) * nthreads, MEM_ALIGNMENT, String("global")));
  
  DBG("Buffers allocated");

  // First, create pipe resources for all threads
  char* next_pipe_start = (char*) pipe_buffer;
  for (int i = 0; i < nthreads; i++) {
    qurt_pipe_attr_t pipe_attr;
    qurt_pipe_attr_init(&pipe_attr);
    qurt_pipe_attr_set_buffer(&pipe_attr, (qurt_pipe_data_t*) next_pipe_start);
    next_pipe_start += thread_pipe_size_bytes;
    qurt_pipe_attr_set_buffer_partition(&pipe_attr, QURT_PIPE_ATTR_MEM_PARTITION_RAM);
    qurt_pipe_attr_set_elements(&pipe_attr, thread_pipe_size_words);

    // create the pipe
    int rc = qurt_pipe_init(&(pipes[i]), &pipe_attr);
    CHECK_EQ(rc, QURT_EOK);
  }

  DBG("Pipes created");

  // Create all threads
  char* next_stack_start = (char*) stack_buffer;
  for (int i = 0; i < nthreads; i++) {
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
    contexts[i] = new ThreadContext(this, i);
    int rc = qurt_thread_create(&(threads[i]), &thread_attr, thread_main, (void*)contexts[i]);
    CHECK_EQ(rc, QURT_EOK);
  }

  DBG("Threads created");
}

void HexagonThreadManager::GetStreamHandles(std::vector<TVMStreamHandle>* out) {
  for (int i = 0; i < nthreads; i++) {
    out->push_back((TVMStreamHandle)i);  // threads identified by index into `threads` array
  }
}
  
/*
PreallocateSyncs is not necessary, but can eliminate runtime overhead for semaphore allocation and vector resizing.
*/
void HexagonThreadManager::PreallocateSyncs(unsigned number_syncs) {
  check_semaphore(number_syncs);
}
  
bool HexagonThreadManager::Dispatch(TVMStreamHandle thread, PackedFunc f, TVMArgs args, TVMRetValue* rv) {
  WrappedPackedFunc* wrapped = new WrappedPackedFunc(f, args, rv);  // WrappedPackedFunc object freed by receiving thread
  return Dispatch(thread, thread_unpack, (void*)wrapped);
}
  
bool HexagonThreadManager::Dispatch(TVMStreamHandle stream, voidfunc f, void* args) {
  unsigned thread = (unsigned)stream;
  DBG("Dispatching to stream " << std::to_string(thread));
  Command* cmd = new Command(f, args);  // Command object freed by receiving thread
  qurt_pipe_data_t msg = (qurt_pipe_data_t)(cmd);
  qurt_pipe_t* pipeAddr = &pipes[thread];

  int trysend = qurt_pipe_try_send(pipeAddr, msg);
  //CHECK_EQ(trysend, 0);
  return trysend == 0;
}

void HexagonThreadManager::Start() {
  thread_signal(&start_semaphore);
}

void HexagonThreadManager::WaitOnThreads() {
  // Using standard signal mechanism to block the "main" thread on all worker threads.
  // Note: this would be slightly more efficient as a barrier, but would need some extra code to
  //  wait on the barrier that would only be used once.
  
  // In case Start() was never explicitly called, call it now to prevent deadlock
  if (qurt_sem_get_val(&start_semaphore) == 0) {
    Start();
  }

  std::vector<qurt_sem_t> finished;
  finished.resize(nthreads);
  
  // initialize one semaphore for each thread
  for (int i = 0; i < nthreads; i++) {
    qurt_sem_init_val(&finished[i], 0);
  }
  // dispatch signal() command to each thread on their private semaphore
  for (int i = 0; i < nthreads; i++) {
    while(!Dispatch((TVMStreamHandle)i, thread_signal, &finished[i]));
  }
  // wait on each semaphore, one at a time
  for (int i = 0; i < nthreads; i++) {
    thread_wait(&finished[i]);
  }
  
  // clean up
  for (int i = 0; i < nthreads; i++) {
    qurt_sem_destroy(&finished[i]);
  }
}
  
void HexagonThreadManager::check_semaphore(unsigned syncID) {
  // extend the semaphore vector if it's not long enough
  if (syncID >= semaphores.size()) {
    auto oldsize = semaphores.size();
    semaphores.resize(syncID + 1);
    for (int i = oldsize; i < semaphores.size(); i++) {
      qurt_sem_init_val(&semaphores[i], 0);
    }
  }
}
  
bool HexagonThreadManager::Signal(TVMStreamHandle thread, SyncPoint syncID) {
  check_semaphore(syncID);
  return Dispatch(thread, thread_signal, (void*) &semaphores[syncID]);
}

bool HexagonThreadManager::Wait(TVMStreamHandle thread, SyncPoint syncID) {
  check_semaphore(syncID);
  return Dispatch(thread, thread_wait, (void*) &semaphores[syncID]);
}

/* Create a sync_from_to relationship with a dynamic semaphore allocation.
Makes use of thread_wait_free to also free the semaphore after sync is complete.
*/
bool HexagonThreadManager::SyncFromTo(TVMStreamHandle signal_thread, TVMStreamHandle wait_thread) {
  qurt_sem_t* sem = (qurt_sem_t*) malloc(sizeof(qurt_sem_t));
  qurt_sem_init_val(sem, 0);
  if(Dispatch(signal_thread, thread_signal, (void*)sem)) {
    return Dispatch(wait_thread, thread_wait_free, (void*)sem);
  } else {
    return false;
  }
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
  DBG("thread exiting");
  qurt_thread_exit( (int) status);
}

void HexagonThreadManager::thread_unpack(void* wpf) {
  WrappedPackedFunc* wrapped = static_cast<WrappedPackedFunc*>(wpf);
  PackedFunc f = wrapped->f;
  TVMArgs args = wrapped->args;
  TVMRetValue* rv = wrapped->rv;
  delete wrapped;  // reclaim memory before call in case call is thread_exit
  f->CallPacked(args, rv);
}
  
void HexagonThreadManager::thread_main(void* context) {
  ThreadContext* tc = static_cast<ThreadContext*>(context);
  unsigned index = tc->index;
  qurt_pipe_t* mypipe = &(tc->tm->pipes[index]);

  DBG("Thread " << std::to_string(index) << " spawned");
   
  while (true) {  // loop, executing commands from pipe
    DBG("Thread " << std::to_string(index) << " receiving command");
    qurt_pipe_data_t msg = qurt_pipe_receive(mypipe);  // blocks if empty
    Command* cmd = (Command*)(msg);
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

#endif  // __hexagon__
