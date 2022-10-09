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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hexagon_buffer.h"
#include "hexagon_buffer_manager.h"
#include "hexagon_thread_manager.h"
#include "hexagon_user_dma.h"
#include "hexagon_vtcm_pool.h"

namespace tvm {
namespace runtime {
namespace hexagon {

/*!
 * \brief Hexagon Device API that is compiled and run on Hexagon.
 */
class HexagonDeviceAPI final : public DeviceAPI {
 public:
  //! \brief Retrieve the global singleton instance of the HexagonDeviceAPI.
  static HexagonDeviceAPI* Global();

  //! \brief Constructor
  HexagonDeviceAPI() { mgr = &hexbuffs; }

  //! \brief Destructor
  ~HexagonDeviceAPI() {}

  //! \brief Ensures resource managers are in a good state for the runtime
  void AcquireResources() {
    CHECK_EQ(runtime_vtcm, nullptr);
    runtime_vtcm = std::make_unique<HexagonVtcmPool>();

    CHECK_EQ(runtime_hexbuffs, nullptr);
    runtime_hexbuffs = std::make_unique<HexagonBufferManager>();
    mgr = runtime_hexbuffs.get();

    CHECK_EQ(runtime_threads, nullptr);
    runtime_threads = std::make_unique<HexagonThreadManager>(threads, stack_size, pipe_size);

    CHECK_EQ(runtime_dma, nullptr);
    runtime_dma = std::make_unique<HexagonUserDMA>();
  }

  //! \brief Ensures all runtime resources are freed
  void ReleaseResources() {
    CHECK(runtime_dma) << "runtime_dma was not created in AcquireResources";
    runtime_dma.reset();

    CHECK(runtime_threads) << "runtime_threads was not created in AcquireResources";
    runtime_threads.reset();

    CHECK(runtime_hexbuffs) << "runtime_hexbuffs was not created in AcquireResources";
    if (runtime_hexbuffs && !runtime_hexbuffs->empty()) {
      LOG(INFO) << "runtime_hexbuffs was not empty in ReleaseResources";
    }
    mgr = &hexbuffs;
    runtime_hexbuffs.reset();

    CHECK(runtime_vtcm) << "runtime_vtcm was not created in AcquireResources";
    runtime_vtcm.reset();
  }

  /*! \brief Currently unimplemented interface to specify the active
   *  Hexagon device.
   */
  void SetDevice(Device dev) final{};

  //! \brief Return the queried Hexagon device attribute.
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final;

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

  //! Erase from HexagonBufferManager and free
  void FreeWorkspace(Device dev, void* data) final;

  /*!
   * \brief Allocate an Nd data space on device with memory scope support.
   *
   * If mem_scope is undefined or is "global", treat shape as the
   * tensor shape, to be flattened into an allocation of 1-d physical
   * memory.  This is done to maintain the semantics expected by callers of
   * DeviceAPI::AllocDataSpace, in cases where it has a valid return value.
   *
   * For other values of mem_scope, the shape is the N-d physical
   * shape of the allocation.
   *
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
   * \brief Allocate an Nd VTCM workspace.
   * \param dev The device to perform the operation.
   * \param ndim The number of dimensions of allocated tensor.
   * \param shape The shape of allocated tensor.
   * \param dtype The element type.
   * \return The allocated HexagonBuffer pointer.
   */
  void* AllocVtcmWorkspace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                           Optional<String> mem_scope);

  //! \brief Free the allocated Nd VTCM workspace.
  void FreeVtcmWorkspace(Device dev, void* ptr);

  /*!
   * \brief Copy data from one storage to another.
   * \note This API is designed to support special memory with shape dependent layout.
   *       DLTensor's are passed with shape information to support these cases.
   * \param from The source array.
   * \param to The target array.
   * \param stream Optional stream object.
   */
  void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) final;

  HexagonThreadManager* ThreadManager() {
    CHECK(runtime_threads) << "runtime_threads has not been created";
    return runtime_threads.get();
  }

  HexagonUserDMA* UserDMA() {
    CHECK(runtime_dma) << "runtime_dma has not been created";
    return runtime_dma.get();
  }

  HexagonVtcmPool* VtcmPool() {
    CHECK(runtime_vtcm) << "runtime_vtcm has not been created";
    return runtime_vtcm.get();
  }

 protected:
  //! Standard Device API interface to copy data from one storage to another.
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final;

 private:
  /*! \brief Helper to check if the device type is valid for the Hexagon Device API
   *  \return Boolean indicating whether the device type is valid
   */
  bool IsValidDevice(DLDevice dev) {
    // Added kDLCPU since we use hexagon as a sub-target of LLVM which by default maps to kDLCPU
    return (TVMDeviceExtType(dev.device_type) == kDLHexagon) ||
           (DLDeviceType(dev.device_type) == kDLCPU);
  }

  //! \brief Manages underlying HexagonBuffer allocations
  // runtime_hexbuffs is used for runtime allocations.  It is created
  // with a call to AcquireResources, and destroyed on ReleaseResources.
  // hexbuffs is used for all allocations outside of the session lifetime.
  HexagonBufferManager hexbuffs;
  std::unique_ptr<HexagonBufferManager> runtime_hexbuffs;

  //! \brief Current buffer manager
  HexagonBufferManager* mgr;

  //! \brief Thread manager
  std::unique_ptr<HexagonThreadManager> runtime_threads;
  const unsigned threads{6};
  const unsigned pipe_size{1000};
  const unsigned stack_size{0x4000};  // 16KB

  //! \brief User DMA manager
  std::unique_ptr<HexagonUserDMA> runtime_dma;

  //! \brief VTCM memory manager
  std::unique_ptr<HexagonVtcmPool> runtime_vtcm;
};
}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_DEVICE_API_H_
