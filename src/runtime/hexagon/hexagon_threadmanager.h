
#ifndef TVM_RUNTIME_HEXAGON_THREADMANAGER
#define TVM_RUNTIME_HEXAGON_THREADMANAGER

// Qualcom lib
#if defined(__hexagon__)
#include "qurt.h"
#endif

// TVM libs
#include "tvm/runtime/logging.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/c_runtime_api.h"
#include "hexagon_common.h"
#include "hexagon_buffer.h"

namespace tvm {
namespace runtime {
namespace hexagon {    

#define DBG(msg) DLOG(INFO) << msg << "\n"

// TODO: adjust as necessary
#define MIN_STACK_SIZE_BYTES 0x4000 // 16KB
#define MAX_STACK_SIZE_BYTES 0x10000 // 64KB 
#define MIN_PIPE_SIZE_WORDS 0x400 // 1K words
#define MAX_PIPE_SIZE_WORDS 0x1000 // 4K words
  
class HexagonThreadManager {
  typedef void (*voidfunc)(void*);
  typedef unsigned SyncPoint;
  const unsigned MEM_ALIGNMENT = 32;
public:
  HexagonThreadManager(unsigned num_threads, unsigned thread_stack_size_bytes, unsigned thread_pipe_size_words);
  ~HexagonThreadManager();
  void GetStreamHandles(std::vector<TVMStreamHandle>* out);
  //void GetThreadHandles(std::vector<void*>* out);
  void PreallocateSyncs(unsigned number_syncs);
  void Dispatch(TVMStreamHandle thread, voidfunc f, void* args);
  void Dispatch(TVMStreamHandle thread, PackedFunc f, TVMArgs args, TVMRetValue* rv = NULL);
  void Signal(TVMStreamHandle thread, SyncPoint syncID);
  void Wait(TVMStreamHandle thread, SyncPoint syncID);
  void SyncFromTo(TVMStreamHandle signal_thread, TVMStreamHandle wait_thread);
  void Start(); // Unblock threads to start execution
  void WaitOnThreads();  // Blocking call to wait until all threads have empty queues
  
private:
  struct ThreadContext {
    HexagonThreadManager* tm;
    unsigned index;
    ThreadContext(HexagonThreadManager* tm, unsigned index) : tm(tm), index(index) {}; 
  };
  
  void SpawnThreads(unsigned thread_stack_size_bytes, unsigned thread_pipe_size_words);
  void check_semaphore(unsigned syncID);
  static void thread_signal(void* semaphore);
  static void thread_wait(void* semaphore);
  static void thread_wait_free(void* semaphore);
  static void thread_exit(void* status);
  static void thread_unpack(void* wrapped);
  static void thread_main(void* context);
  unsigned nthreads;
  void* stack_buffer;
  void* pipe_buffer;
  #if defined(__hexagon__)
  qurt_thread_t* threads;
  qurt_pipe_t* pipes;
  ThreadContext** contexts;
  std::vector<qurt_sem_t> semaphores;
  qurt_sem_t start_semaphore;
  #endif

  /*
    Encapsulate a function pointer + arg pointer. Sent via pipe to threads to execute.
    Function should have type   "void myfunc(void* args)"
  */
  struct Command {
    voidfunc f;
    void* args;
    Command(voidfunc f, void* args) : f(f), args(args) {};
  };

  /*
    Encapsulate a PackedFunc + args + return value pointer.
    Used to wrap a PackedFunc call into a single object for use in a Command.
   */
  struct WrappedPackedFunc {
    PackedFunc f;
    TVMArgs args;
    TVMRetValue* rv;
    WrappedPackedFunc(PackedFunc f, TVMArgs args, TVMRetValue* rv) : f(f), args(args), rv(rv) {};
  };

};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

//#endif  // __hexagon__
#endif  // TVM_RUNTIME_HEXAGON_THREADMANAGER
