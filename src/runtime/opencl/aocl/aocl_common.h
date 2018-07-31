/*!
 *  Copyright (c) 2018 by Contributors
 * \file aocl_common.h
 * \brief AOCL common header
 */
#ifndef TVM_RUNTIME_OPENCL_AOCL_AOCL_COMMON_H_
#define TVM_RUNTIME_OPENCL_AOCL_AOCL_COMMON_H_

#include "../opencl_common.h"

namespace tvm {
namespace runtime {
namespace cl {

/*!
 * \brief Process global AOCL workspace.
 */
class AOCLWorkspace final : public OpenCLWorkspace {
 public:
  // override OpenCL device API
  void Init() final;
  bool IsOpenCLDevice(TVMContext ctx) final;
  OpenCLThreadEntry* GetThreadEntry() final;
  // get the global workspace
  static const std::shared_ptr<OpenCLWorkspace>& Global();
};


/*! \brief Thread local workspace for AOCL */
class AOCLThreadEntry : public OpenCLThreadEntry {
 public:
  // constructor
  AOCLThreadEntry()
      : OpenCLThreadEntry(static_cast<DLDeviceType>(kDLAOCL), AOCLWorkspace::Global()) {}

  // get the global workspace
  static AOCLThreadEntry* ThreadLocal();
};
}  // namespace cl
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENCL_AOCL_AOCL_COMMON_H_
