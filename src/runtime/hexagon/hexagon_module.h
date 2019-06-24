/*!
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
 * This Contribution is being provided by Qualcomm Technologies, Inc.,
 * a Delaware corporation, or its subsidiary Qualcomm Innovation Center, Inc.,
 * a California corporation, under certain additional terms and conditions
 * pursuant to Section 5 of the Apache 2.0 license.  In this regard, with
 * respect to this Contribution, the term "Work" in Section 1 of the
 * Apache 2.0 license means only the specific subdirectory within the TVM repo
 * (currently at https://github.com/dmlc/tvm) to which this Contribution is
 * made.
 * In any case, this submission is "Not a Contribution" with respect to its
 * permitted use with any of the "vta" and "verilog" subdirectories in the TVM
 * repo.
 * Qualcomm Technologies, Inc. and Qualcomm Innovation Center, Inc. retain
 * copyright of their respective Contributions.
 */
#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_MODULE_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_MODULE_H_

#include <dmlc/logging.h>
#include <tvm/runtime/module.h>

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

#include "../meta_data.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Create a Hexagon module from data.
 * \param data  The module data.
 * \param fmt   The format of the data, can be "obj".
 * \param fmap  The map function information map of each function.
 */
Module HexagonModuleCreate(std::string data, std::string fmt,
                           std::unordered_map<std::string, FunctionInfo> fmap);

namespace hexagon {

/*!
 * \brief Low-level interface for communicating with Hexagon devices.
 */
class Device {
 public:
  /*!
   * \brief Allocate memory on device.
   * \param size    Requested size.
   * \param align   Requested alignment.
   * \return        Pointer (local to the device) of the allocated memory,
   *                or nullptr if allocation failed.
   */
  virtual void* Alloc(unsigned size, unsigned align) = 0;
  /*!
   * \brief Release allocated memory on device.
   * \param ptr     Pointer to memory previously allocated by \ref Alloc.
   */
  virtual void Free(void* ptr) = 0;
  /*!
   * \brief Copy a block of data on device to another location on the device.
   * \param dst     Pointer (local to device) to the destination buffer.
   * \param src     Pointer (local to device) of the source buffer.
   * \param len     Number of bytes to copy.
   */
  virtual void CopyDeviceToDevice(void* dst, const void* src,
                                  unsigned len) = 0;
  /*!
   * \brief Copy a block of data from device to host.
   * \param host_dst  Pointer (local to host) to the destination buffer.
   * \param src       Pointer (local to device) to the source buffer.
   * \param len       Number of bytes to copy.
   */
  virtual void CopyDeviceToHost(void* host_dst, const void* src,
                                unsigned len) = 0;
  /*!
   * \brief Copy a block of data from host to device.
   * \param dst       Pointer (local to device) to the destination buffer.
   * \param host_src  Pointer (local to host) to the source buffer.
   * \param len       Number of bytes to copy.
   */
  virtual void CopyHostToDevice(void* dst, const void* host_src,
                                unsigned len) = 0;
  /*!
   * \brief Load a module (typically a shared library) into device.
   * \param data    Name of the shared library.
   * \param fmt     Format of the library (currently ignored).
   * \return        Pointer to the loaded module.
   * \note Currently only one module can be loaded at any given time.
   */
  virtual void* Load(const std::string& data, const std::string& fmt) = 0;
  /*!
   * \brief Unload a module from device.
   * \param mod     Pointer to a loaded module returned by \ref Load.
   */
  virtual void Unload(void* mod) = 0;
  /*!
   * \brief Find the address of an object in the currently loaded module.
   * \param sym     Name of the object.
   * \return Address of the located object, or nullptr if object was
   *         not found.
   */
  virtual void* Resolve(const std::string& sym) = 0;
  /*!
   * \brief Invoke a function on device with given arguments.
   * \param func    Address (local to device) of the function to call.
   * \param scalar  Pointer to an array of 32-bit values that will be
   *                passed via consecutive registers: r0..r5. This array
   *                includes dummy values for skipped registers.
   * \param sc_num  Number of values in the "scalar" array.
   * \param stack   Pointer to an array of 32-bit values that will be
   *                passed on the stack. This array includes dummy values
   *                for padding.
   * \param st_num  Number of values in the "stack" array.
   */
  virtual void Call(void* func, uint32_t* scalar, unsigned sc_num,
                    uint32_t* stack, unsigned st_num) = 0;

  virtual ~Device() = 0;

  static Device* Global();
  static bool ValidateDeviceId(decltype(DLContext::device_id) device_id) {
    // Only supporting a single device for now.
    return device_id == 0;
  }
};

}  // namespace hexagon

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_MODULE_H_
