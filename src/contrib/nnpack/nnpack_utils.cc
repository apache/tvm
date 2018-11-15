/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external nnpack library call.
 */
#include "nnpack_utils.h"

namespace tvm {
namespace contrib {
using namespace runtime;

typedef dmlc::ThreadLocalStore<NNPackThreadLocalEntry> NNPackThreadLocalStore;


NNPackThreadLocalEntry* NNPackThreadLocalEntry::ThreadLocal() {
  return NNPackThreadLocalStore::Get();
}

bool NNPackConfig(uint64_t nthreads) {
  NNPackThreadLocalEntry *entry = NNPackThreadLocalEntry::ThreadLocal();
  if (entry->threadpool && pthreadpool_get_threads_count(entry->threadpool) == nthreads) {
    CHECK_NE(nthreads, 1);
    return true;
  }
  if (entry->threadpool) {
    pthreadpool_destroy(entry->threadpool);
    entry->threadpool = nullptr;
  }

  if (nthreads == 1) {
    // a null threadpool means the function is invoked on the calling thread,
    // which is the desired logic for nthreads == 1
    CHECK(!entry->threadpool);
    return true;
  }

  entry->threadpool = pthreadpool_create(nthreads);
  return true;
}


TVM_REGISTER_GLOBAL("contrib.nnpack._initialize")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = nnp_initialize();
  });

}  // namespace contrib
}  // namespace tvm
