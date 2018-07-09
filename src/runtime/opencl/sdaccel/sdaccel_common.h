/*!
 *  Copyright (c) 2018 by Contributors
 * \file sdaccel_common.h
 * \brief SDAccel common header
 */
#ifndef TVM_RUNTIME_OPENCL_SDACCEL_SDACCEL_COMMON_H_
#define TVM_RUNTIME_OPENCL_SDACCEL_SDACCEL_COMMON_H_

#include "../opencl_common.h"

namespace tvm {
namespace runtime {
namespace cl {

/*!
 * \brief Process global SDAccel workspace.
 */
class SDAccelWorkspace final : public OpenCLWorkspace {
 public:
  // override OpenCL device API
  void Init() final;
  bool IsOpenCLDevice(TVMContext ctx) final;
  OpenCLThreadEntry* GetThreadEntry() final;
  // get the global workspace
  static const std::shared_ptr<OpenCLWorkspace>& Global();
};


/*! \brief Thread local workspace for SDAccel*/
class SDAccelThreadEntry : public OpenCLThreadEntry {
 public:
  // constructor
  SDAccelThreadEntry()
      : OpenCLThreadEntry(static_cast<DLDeviceType>(kDLSDAccel), SDAccelWorkspace::Global()) {}

  // get the global workspace
  static SDAccelThreadEntry* ThreadLocal();
};
}  // namespace cl
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENCL_SDACCEL_SDACCEL_COMMON_H_
