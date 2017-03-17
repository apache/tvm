/*!
 *  Copyright (c) 2016 by Contributors
 * \file device_api.h
 * \brief Device specific API
 */
#ifndef TVM_RUNTIME_DEVICE_API_H_
#define TVM_RUNTIME_DEVICE_API_H_

#include <tvm/base.h>
#include <tvm/runtime/c_runtime_api.h>
#include <string>

namespace tvm {
namespace runtime {

class DeviceAPI {
 public:
  /*! \brief virtual destructor */
  virtual ~DeviceAPI() {}
  /*!
   * \brief Allocate a data space on device.
   * \param ctx The device context to perform operation.
   * \param size The size of the memory
   * \param alignment The alignment of the memory.
   * \return The allocated device pointer
   */
  virtual void* AllocDataSpace(TVMContext ctx, size_t size, size_t alignment) = 0;
  /*!
   * \brief Free a data space on device.
   * \param ctx The device context to perform operation.
   * \param ptr The data space.
   * \tparam xpu The device mask.
   */
  virtual void FreeDataSpace(TVMContext ctx, void* ptr) = 0;
  /*!
   * \brief copy data from one place to another
   * \param dev The device to perform operation.
   * \param from The source array.
   * \param to The target array.
   * \param size The size of the memory
   * \param ctx_from The source context
   * \param ctx_to The target context
   */
  virtual void CopyDataFromTo(const void* from,
                              void* to,
                              size_t size,
                              TVMContext ctx_from,
                              TVMContext ctx_to,
                              TVMStreamHandle stream) = 0;
  /*!
   * \brief Synchronize the stream
   * \param ctx The context to perform operation.
   * \param stream The stream to be sync.
   */
  virtual void StreamSync(TVMContext ctx, TVMStreamHandle stream) = 0;
};

/*!
 * \brief The name of Device API factory.
 * \param type The device type.
 */
inline std::string DeviceName(DLDeviceType type) {
  switch (static_cast<int>(type)) {
    case kCPU: return "cpu";
    case kGPU: return "gpu";
    case kOpenCL: return "opencl";
    case kVPI: return "vpi";
    default: LOG(FATAL) << "unknown type =" << type; return "Unknown";
  }
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DEVICE_API_H_
