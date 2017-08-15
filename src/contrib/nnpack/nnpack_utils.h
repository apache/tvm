/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external nnpack library call.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/thread_local.h>
#include <dmlc/logging.h>
#include <nnpack.h>

namespace tvm {
namespace contrib {
using namespace runtime;

struct NNPackThreadLocalEntry {
  pthreadpool_t threadpool{NULL};
};

typedef dmlc::ThreadLocalStore<NNPackThreadLocalEntry> NNPackThreadLocalStore;
}  // namespace contrib
}  // namespace tvm
