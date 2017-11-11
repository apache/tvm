/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external nnpack library call.
 */
#include "./nnpack_utils.h"

namespace tvm {
namespace contrib {
using namespace runtime;

typedef dmlc::ThreadLocalStore<NNPackThreadLocalEntry> NNPackThreadLocalStore;

NNPackThreadLocalEntry* NNPackThreadLocalEntry::ThreadLocal() {
  return NNPackThreadLocalStore::Get();
}

bool NNPackConfig(uint64_t nthreads) {
  NNPackThreadLocalEntry *entry = NNPackThreadLocalEntry::ThreadLocal();
  if (entry->threadpool != NULL &&
      pthreadpool_get_threads_count(entry->threadpool) != nthreads) {
    pthreadpool_destroy(entry->threadpool);
    entry->threadpool = NULL;
  }
  if (entry->threadpool == NULL) {
    entry->threadpool = pthreadpool_create(nthreads);
  }
  return true;
}


TVM_REGISTER_GLOBAL("contrib.nnpack._Config")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    CHECK(NNPackConfig(args[0]));
  });
}  // namespace contrib
}  // namespace tvm
