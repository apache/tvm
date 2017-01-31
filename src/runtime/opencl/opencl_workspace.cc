/*!
 *  Copyright (c) 2017 by Contributors
 * \file opencl_workspace.cc
 */
#include "./opencl_common.h"

#if TVM_OPENCL_RUNTIME

#include <dmlc/thread_local.h>

namespace tvm {
namespace runtime {
namespace cl {

OpenCLWorkspace* OpenCLWorkspace::Global() {
  static OpenCLWorkspace inst;
  return &inst;
}

typedef dmlc::ThreadLocalStore<OpenCLThreadEntry> OpenCLThreadStore;

OpenCLThreadEntry* OpenCLThreadEntry::ThreadLocal() {
  return OpenCLThreadStore::Get();
}

}  // namespace cl
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENCL_RUNTIME
