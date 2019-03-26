/*!
 *  Copyright (c) 2019 by Contributors
 * \file host_low_level_device.cc
 * \brief emulated low-level micro device implementation on host machine
 */

#include "low_level_device.h"

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
  HostLowLevelDevice(size_t num_bytes);

  /*!
   * \brief destructor to deallocate on-host device region
   */
  ~HostLowLevelDevice();

  void Write(void* offset,
             void* buf,
             size_t num_bytes) final;

  void Read(void* offset,
            void* buf,
            size_t num_bytes) final;

  void Execute(void* func_addr, void* breakpoint) final;

  const void* base_addr() const final;

 private:
  /*! \brief base address of the micro device memory region */
  void* base_addr_;
  /*! \brief size of memory region */
  size_t size_;
};

const std::shared_ptr<LowLevelDevice> HostLowLevelDeviceCreate(size_t num_bytes) {
  return nullptr;
}
} // namespace runtime
} // namespace tvm
