/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file low_level_device.h
 * \brief Abstract low-level micro device management
 */
#ifndef TVM_RUNTIME_MICRO_LOW_LEVEL_DEVICE_H_
#define TVM_RUNTIME_MICRO_LOW_LEVEL_DEVICE_H_

#include <memory>
#include <string>

#include "micro_common.h"

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
   * \brief reads num_bytes from device memory at base_addr + offset into buffer
   * \param offset on-device memory offset pointer to be read from
   * \param buffer on-host buffer to be read into
   * \param num_bytes number of bytes to be read
   */
  virtual void Read(DevBaseOffset offset,
                    void* buffer,
                    size_t num_bytes) = 0;

  /*!
   * \brief writes num_bytes from buffer to device memory at base_addr + offset
   * \param offset on-device memory offset pointer to be written to
   * \param buffer on-host buffer to be written
   * \param num_bytes number of bytes to be written
   */
  virtual void Write(DevBaseOffset offset,
                     const void* buffer,
                     size_t num_bytes) = 0;

  /*!
   * \brief starts execution of device at offset
   * \param func_addr offset of the init stub function
   * \param breakpoint breakpoint at which to stop function execution
   */
  virtual void Execute(DevBaseOffset func_offset, DevBaseOffset breakpoint) = 0;

  // TODO(weberlo): Should we just give the device the *entire* memory layout
  // decided by the session?

  /*!
   * \brief sets the offset of the top of the stack section
   * \param stack_top offset of the stack top
   */
  virtual void SetStackTop(DevBaseOffset stack_top) {
    LOG(FATAL) << "unimplemented";
  }

  /*!
   * \brief convert from base offset to absolute address
   * \param offset base offset
   * \return absolute address
   */
  DevPtr ToDevPtr(DevBaseOffset offset) {
    return DevPtr(base_addr() + offset.value());
  }

  /*!
   * \brief convert from absolute address to base offset
   * \param ptr absolute address
   * \return base offset
   */
  DevBaseOffset ToDevOffset(DevPtr ptr) {
    return DevBaseOffset(ptr.value() - base_addr());
  }

  /*!
   * \brief getter function for low-level device type
   * \return string containing device type
   */
  virtual const char* device_type() const = 0;

 protected:
  /*!
   * \brief getter function for base_addr
   * \return the base address of the device memory region
   */
  virtual std::uintptr_t base_addr() const = 0;
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
const std::shared_ptr<LowLevelDevice> OpenOCDLowLevelDeviceCreate(std::uintptr_t base_addr,
                                                                  const std::string& addr,
                                                                  int port);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_LOW_LEVEL_DEVICE_H_
