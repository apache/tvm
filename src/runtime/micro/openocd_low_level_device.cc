/*!
 *  Copyright (c) 2019 by Contributors
 * \file openocd_low_level_device.cc
 * \brief openocd low-level device to interface with micro devices over JTAG
 */

#include "low_level_device.h"

namespace tvm {
namespace runtime {

// TODO(weberlo): Add implementation for this device.

/*!
 * \brief openocd low-level device for uTVM micro devices connected over JTAG
 */
class OpenOCDLowLevelDevice final : public LowLevelDevice {
 public:
  /*!
   * \brief constructor to initialize connection to openocd device
   * \param port port of the OpenOCD server to connect to
   */
  explicit OpenOCDLowLevelDevice(int port);

  /*!
   * \brief destructor to close openocd device connection
   */
  ~OpenOCDLowLevelDevice();

  void Write(DevBaseOffset offset,
             void* buf,
             size_t num_bytes) final;

  void Read(DevBaseOffset offset,
            void* buf,
            size_t num_bytes) final;

  void Execute(DevBaseOffset func_addr, DevBaseOffset breakpoint) final;

  DevBaseAddr base_addr() const final;

  const char* device_type() const final {
    return "openocd";
  }

 private:
  /*! \brief base address of the micro device memory region */
  DevBaseAddr base_addr_;
  /*! \brief size of memory region */
  size_t size_;
};

const std::shared_ptr<LowLevelDevice> OpenOCDLowLevelDeviceCreate(int port) {
  return nullptr;
}
}  // namespace runtime
}  // namespace tvm
