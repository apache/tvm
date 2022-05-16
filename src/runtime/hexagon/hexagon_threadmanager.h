
#ifndef TVM_RUNTIME_HEXAGON_THREADMANAGER
#define TVM_RUNTIME_HEXAGON_THREADMANAGER

#if defined(__hexagon__)

// Qualcom lib
#include "qurt.h"

// TVM libs
#include "tvm/runtime/logging.h"
#include "hexagon_common.h"
#include "hexagon_buffer.h"

namespace tvm {
namespace runtime {
namespace hexagon {    
  
class HexagonThreadManager {
  typedef void (*voidfunc)(void*);
  const unsigned MEM_ALIGNMENT = 32;
public:
  HexagonThreadManager(unsigned num_threads, unsigned thread_stack_size_bytes, unsigned thread_pipe_size_words);
  ~HexagonThreadManager();
  void GetStreamHandles(std::vector<TVMStreamHandle>* out);
  void AllocateSyncs(unsigned number_syncs);
  void Dispatch(unsigned thread, voidfunc f, void* args);
  void Signal(unsigned thread, unsigned syncID);
  void Wait(unsigned thread, unsigned syncID);
  void SyncFromTo(unsigned signal_thread, unsigned wait_thread);
  
private:
  struct ThreadContext {
    HexagonThreadManager* tm;
    unsigned index;
    ThreadContext(HexagonThreadManager* tm, unsigned index) { this->tm = tm; this->index = index; };
  };
  
  void SpawnThreads();
  static void thread_signal(void* semaphore);
  static void thread_wait(void* semaphore);
  static void thread_wait_free(void* semaphore);
  static void thread_exit(void* status);
  static void thread_main(void* context);
  unsigned nthreads;
  void* stack_buffer;
  void* pipe_buffer;
  unsigned thread_stack_size;  //bytes
  unsigned thread_pipe_size;  // bytes
  unsigned thread_pipe_size_words;  // words (qurt_pipe_data_t, 64 bit on this platform)
  qurt_thread_t* threads;
  qurt_pipe_t* pipes;
  ThreadContext** contexts;
  unsigned number_semaphores;
  qurt_sem_t* semaphores;
  
  class Command {
  public:
    voidfunc f;
    void* args;
    Command(voidfunc f, void* args) { this->f = f; this->args = args; }
  };

};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // __hexagon__
#endif  // TVM_RUNTIME_HEXAGON_THREADMANAGER
