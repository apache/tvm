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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_DEVICE_API_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_DEVICE_API_H_

#include <tvm/runtime/device_api.h>

namespace tvm {
namespace runtime {
namespace hexagon {

class HexagonBuffer;

/*!
 * \brief Hexagon Device API that is compiled and run on Hexagon.
 */
class HexagonDeviceAPIv2 final : public DeviceAPI {
 public:
  //! \brief Retrieve the global singleton instance of the HexagonDeviceAPIv2.
  static HexagonDeviceAPIv2* Global();

  //! \brief Constructor
  HexagonDeviceAPIv2() {}

  //! \brief Destructor
  ~HexagonDeviceAPIv2() {}

  /*! \brief Currently unimplemented interface to specify the active
   *  Hexagon device.
   */
  void SetDevice(Device dev) final {};

  //! \brief Return the queried Hexagon device attribute.
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue * rv) final;

  //! \brief Currently unimplemented interface to synchronize a device stream.
  void StreamSync(Device dev, TVMStreamHandle stream) final {}

  //! \note Standard memory allocation methods of the DeviceAPI interface.
  //! \brief Allocate a flat allocation of global memory wrapped in a HexagonBuffer.
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final;

  //! \brief Free the allocated HexagonBuffer.
  void FreeDataSpace(Device dev, void* ptr) final;

  /*! \brief Request a dynamically allocated HexagonBuffer from a workspace pool.
   *  \returns The underlying allocation pointer.
   */
  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;

  //! Dereference workspace pool and erase from tracked workspace_allocations_.
  void FreeWorkspace(Device dev, void* data) final;

  /*!
   * \brief Allocate an Nd data space on device with memory scope support.
   * \param dev The device to perform the operation.
   * \param ndim The number of dimensions of allocated tensor.
   * \param shape The shape of allocated tensor.
   * \param dtype The element type.
   * \param mem_scope The memory scope of the allocated tensor.
   * \return The allocated HexagonBuffer pointer.
   */
  void* AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                       Optional<String> mem_scope) final;

  /*!
   * \brief Copy data from one storage to another.
   * \note This API is designed to support special memory with shape dependent layout.
   *       DLTensor's are passed with shape information to support these cases.
   * \param from The source array.
   * \param to The target array.
   * \param stream Optional stream object.
   */
  void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) final;
 protected:
  //! Standard Device API interface to copy data from one storage to another.
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset,
                      size_t size, Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final;
 private:
  //! Lookup table for the HexagonBuffer managing a workspace allocation.
  std::unordered_map<void*, HexagonBuffer*> workspace_allocations_;
};
}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_DEVICE_API_H_
