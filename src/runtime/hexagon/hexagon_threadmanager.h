
#ifndef TVM_RUNTIME_HEXAGON_THREADMANAGER
#define TVM_RUNTIME_HEXAGON_THREADMANAGER

// Qualcom libs
#include "qurt_pipe.h"
#include "qurt_thread.h"
#incdlue "qurt_alloc.h"
#include "qurt_consts.h"


// TVM libs
#include "tvm/runtime/logging.h"
#include "hexgon_buffer.h"

namespace tvm {
namespace runtime {
namespace hexagon {    

// platform constants
#define MEM_ALIGNMENT 32
#define HVX_MODE QURT_HVX_MODE_128B
  
class HexagonThreadManager {
public:
  HexagonThreadManager(unsigned num_threads, unsigned thread_stack_size_bytes, unsigned thread_pipe_size_words);
  ~HexagonThreadManager();
  void GetStreamhandles(std::vector<TVMStreamHandle>* out);
  void AllocateSyncs(unsigned number_syncs);
  void Dispatch(unsigned thread, voidfunc f, void* args);
  void Signal(unsigned thread, unsigned syncID);
  void Wait(unsigned thread, unsigned syncID);
  void SyncFromTo(unsigned signal_thread, unsigned wait_thread);
  
private:
  void SpawnThreads();
  static void thread_signal(void* semaphore);
  static void thread_wait(void* semaphore);
  static void thread_wait_free(void* semaphore);
  static void thread_exit(void* status);
  static void thread_main(void* context);
  unsigned nthreads;
  HexagonBuffer stack_buffer;
  HexagonBuffer pipe_buffer;
  unsigned thread_stack_size;  //bytes
  unsigned thread_pipe_size;  // bytes
  unsigned thread_pipe_size_words;  // words (qurt_pipe_data_t, 64 bit on this platform)
  qurt_thread_t threads[];
  qurt_pipe_t pipes[];
  ThreadContext* contexts[];
  unsigned number_semaphores;
  qurt_sem_t* semaphores;

  typedef void (*)(void*) voidfunc;
  
  class Command {
  private:
    voidfunc f;
    void* args;
  public:
    Command(voidfunc f, void* args) { this->f = f; this->args = args; }
  }

  struct ThreadContext {
    HexagonThreadManager* tm;
    unsigned index;
    ThreadContext(HexagonThreadManager* tm, unsigned index) { this->tm = tm; this->index = index };
  };
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_THREADMANAGER
