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
   * \brief reads num_bytes from device memory at addr into buffer
   * \param addr on-device memory address to read from
   * \param buffer on-host buffer to be read into
   * \param num_bytes number of bytes to read
   */
  virtual void Read(TargetPtr addr,
                    void* buffer,
                    size_t num_bytes) = 0;

  /*!
   * \brief writes num_bytes from buffer to device memory at addr
   * \param addr on-device memory address to write into
   * \param buffer host buffer to write from
   * \param num_bytes number of bytes to write
   */
  virtual void Write(TargetPtr addr,
                     const void* buffer,
                     size_t num_bytes) = 0;

  /*!
   * \brief starts execution of device at func_addr
   * \param func_addr offset of the init stub function
   * \param breakpoint_addr address at which to stop function execution
   */
  virtual void Execute(TargetPtr func_addr, TargetPtr breakpoint_addr) = 0;

  /*!
   * \brief getter function for low-level device type
   * \return string containing device type
   */
  virtual const char* device_type() const = 0;
};

/*!
 * \brief create a host low-level device
 * \param num_bytes size of the memory region
 * \param base_addr pointer to write the host device's resulting base address into
 */
const std::shared_ptr<LowLevelDevice> HostLowLevelDeviceCreate(size_t num_bytes,
                                                               TargetPtr* base_addr);

/*!
 * \brief connect to OpenOCD and create an OpenOCD low-level device
 * \param addr address of the OpenOCD server to connect to
 * \param port port of the OpenOCD server to connect to
 */
const std::shared_ptr<LowLevelDevice> OpenOCDLowLevelDeviceCreate(const std::string& addr,
                                                                  int port);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_LOW_LEVEL_DEVICE_H_
