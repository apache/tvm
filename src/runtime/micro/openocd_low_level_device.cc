/*!
 *  Copyright (c) 2019 by Contributors
 * \file openocd_low_level_device.cc
 * \brief openocd low-level device to interface with micro devices over JTAG
 */

#include "low_level_device.h"

namespace tvm {
namespace runtime {
/*!
 * \brief openocd low-level device for uTVM micro devices connected over JTAG
 */
class OpenOCDLowLevelDevice final : public LowLevelDevice {
 public:
  /*!
   * \brief constructor to initialize connection to openocd device
   * \param port port of the OpenOCD server to connect to
   */
  OpenOCDLowLevelDevice(int port);

  /*!
   * \brief destructor to close openocd device connection
   */
  ~OpenOCDLowLevelDevice();

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

const std::shared_ptr<LowLevelDevice> OpenOCDLowLevelDeviceCreate(int port) {
  return nullptr;
}
} // namespace runtime
} // namespace tvm
