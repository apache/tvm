/*!
 *  Copyright (c) 2019 by Contributors
 * \file low_level_device.h
 * \brief Abstract low-level micro device management
 */
#ifndef TVM_RUNTIME_MICRO_LOW_LEVEL_DEVICE_H_
#define TVM_RUNTIME_MICRO_LOW_LEVEL_DEVICE_H_

#include <cstddef>
#include <memory>

namespace tvm {
namespace runtime {
/*!
 * \brief virtual interface for low-level micro device management
 */
class LowLevelDevice {
 public:
  /*! \brief virtual destructor */
  virtual ~LowLevelDevice() {}

  /*!
   * \brief writes num_bytes from buffer to device memory at base_addr + offset
   * \param offset on-device memory offset pointer to be written to
   * \param buffer on-host buffer to be written
   * \param num_bytes number of bytes to be written
   */
  virtual void Write(void* offset,
                     void* buffer,
                     size_t num_bytes) = 0;

  /*!
   * \brief reads num_bytes from device memory at base_addr + offset into buffer
   * \param offset on-device memory offset pointer to be read from
   * \param buffer on-host buffer to be read into
   * \param num_bytes number of bytes to be read
   */
  virtual void Read(void* offset,
                    void* buffer,
                    size_t num_bytes) = 0;

  /*!
   * \brief starts execution of device at offset
   * \param func_addr address of the init stub function
   * \param breakpoint breakpoint at which to stop function execution
   */
  virtual void Execute(void* func_addr, void* breakpoint) = 0;

  /*!
   * \brief getter function for base_addr
   * \return the base address of the device memory region
   */
  virtual const void* base_addr() const = 0;
};

/*!
 * \brief create a host low-level device
 * \param num_bytes size of the memory region
 */
const std::shared_ptr<LowLevelDevice> HostLowLevelDeviceCreate(size_t num_bytes);

/*!
 * \brief connect to OpenOCD and create an OpenOCD low-level device
 * \param port port of the OpenOCD server to connect to
 */
const std::shared_ptr<LowLevelDevice> OpenOCDLowLevelDeviceCreate(int port);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_LOW_LEVEL_DEVICE_H_
