/*!
 *  Copyright (c) 2019 by Contributors
 * \file host_low_level_device.h
 * \brief emulated low-level micro device implementation on host machine
 */
#ifndef TVM_RUNTIME_MICRO_HOST_LOW_LEVEL_DEVICE_API_H_
#define TVM_RUNTIME_MICRO_HOST_LOW_LEVEL_DEVICE_API_H_

#include <sys/mman.h>
#include <cstring>
#include "low_level_device.h"
#include "micro_common.h"

namespace tvm {
namespace runtime {
/*!
 * \brief emulated low-level device on host machine
 */
class HostLowLevelDevice final : public LowLevelDevice {
 public:
  /*!
   * \brief constructor to initialize on-host memory region to act as device
   * \param num_bytes size of the emulated on-device memory region
   */
  explicit HostLowLevelDevice(size_t num_bytes);

  /*!
   * \brief destructor to deallocate on-host device region
   */
  virtual ~HostLowLevelDevice();

  void Read(DevBaseOffset offset, void* buf, size_t num_bytes) final;

  void Write(DevBaseOffset offset, void* buf, size_t num_bytes) final;

  void Execute(DevBaseOffset func_offset, DevBaseOffset breakpoint) final;

  DevBaseAddr base_addr() const final {
    return base_addr_;
  }

  const char* device_type() const final {
    return "host";
  }

 private:
  /*! \brief base address of the micro device memory region */
  DevBaseAddr base_addr_;
  /*! \brief size of memory region */
  size_t size_;
};

/*!
 * \brief create a host low-level device
 * \param num_bytes size of the memory region
 */
const std::shared_ptr<LowLevelDevice> HostLowLevelDeviceCreate(size_t num_bytes);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_HOST_LOW_LEVEL_DEVICE_API_H_
