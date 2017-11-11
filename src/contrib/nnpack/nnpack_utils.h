/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external nnpack library call.
 */
#ifndef TVM_CONTRIB_NNPACK_NNPACK_UTILS_H_
#define TVM_CONTRIB_NNPACK_NNPACK_UTILS_H_
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
  static NNPackThreadLocalEntry* ThreadLocal();
};

bool NNPackConfig(uint64_t nthreads);
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_NNPACK_NNPACK_UTILS_H_
