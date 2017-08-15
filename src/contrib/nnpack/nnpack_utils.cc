/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external nnpack library call.
 */
#include "./nnpack_utils.h"

namespace tvm {
namespace contrib {
using namespace runtime;

TVM_REGISTER_GLOBAL("contrib.nnpack._Config")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    NNPackThreadLocalEntry *entry = NNPackThreadLocalStore::Get();
    size_t nthreads = args[0];
    if (entry->threadpool != NULL &&
        pthreadpool_get_threads_count(entry->threadpool) != nthreads) {
      pthreadpool_destroy(entry->threadpool);
      entry->threadpool = NULL;
    }
    if (entry->threadpool == NULL) {
      entry->threadpool = pthreadpool_create(nthreads);
    }
  });
}  // namespace contrib
}  // namespace tvm
