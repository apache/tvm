#include "parallel_for.h"

#include <tvm/ir/error.h>

namespace tvm {
namespace support {

ThreadPool& ThreadPool::Global() {
  static ThreadPool* pool = new ThreadPool();
  static int ct = 0;

  ct = (ct + 1) % ThreadPool::REFRESH_EVERY;

  if (ct == 0) {
    pool->Abort();
    delete pool;
    pool = new ThreadPool();
  }

  if (pool->NumWorkers() == 0) {
    pool->Launch(std::thread::hardware_concurrency());
  }

  return *pool;
}

void parallel_for(int begin, int end, const std::function<void(int)>& f, int step) {
  auto& pf = ThreadPool::Global();
  int batch_count = (end - begin) / step;
  CHECK_GT(batch_count, 0);
  pf.BeginBatch(batch_count);
  for (int i = begin; i < end; i += step) {
    pf.Enqueue(f, i);
  }
  pf.WaitBatch();
}

}  // namespace support
}  // namespace tvm
