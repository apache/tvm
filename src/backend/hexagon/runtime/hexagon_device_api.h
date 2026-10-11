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

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../runtime/workspace_pool.h"
#include "hexagon_buffer.h"
#include "hexagon_buffer_manager.h"
#include "hexagon_power_manager.h"
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
  HexagonDeviceAPI() {}

  //! \brief Destructor
  ~HexagonDeviceAPI() {}

  //! \brief Ensures resource managers are in a good state for the runtime
  void AcquireResources() {
    TVM_FFI_ICHECK_EQ(runtime_power_manager, nullptr);
    TVM_FFI_ICHECK_EQ(runtime_vtcm, nullptr);
    auto resources = AcquireDeviceResources();
    runtime_power_manager = std::shared_ptr<HexagonPowerManager>(resources, &resources->power);
    runtime_vtcm = std::shared_ptr<HexagonVtcmPool>(resources, &resources->vtcm);

    TVM_FFI_ICHECK_EQ(runtime_hexbuffs, nullptr);
    runtime_hexbuffs = std::make_shared<HexagonBufferManager>();

    TVM_FFI_ICHECK_EQ(runtime_threads, nullptr);
    runtime_threads =
        std::make_unique<HexagonThreadManager>(threads, stack_size, pipe_size, hw_resources);

    TVM_FFI_ICHECK_EQ(runtime_dma, nullptr);
    runtime_dma = std::make_unique<HexagonUserDMA>();
  }

  //! \brief Ensures all runtime resources are freed
  void ReleaseResources() {
    TVM_FFI_ICHECK(runtime_dma) << "runtime_dma was not created in AcquireResources";
    runtime_dma.reset();

    TVM_FFI_ICHECK(runtime_threads) << "runtime_threads was not created in AcquireResources";
    runtime_threads.reset();

    ResetWorkspacePool(kDLHexagon);
    TVM_FFI_ICHECK(runtime_hexbuffs) << "runtime_hexbuffs was not created in AcquireResources";
    runtime_hexbuffs.reset();

    TVM_FFI_ICHECK(runtime_vtcm) << "runtime_vtcm was not created in AcquireResources";
    runtime_vtcm.reset();

    TVM_FFI_ICHECK(runtime_power_manager)
        << "runtime_power_manager was not created in AcquireResources";
    runtime_power_manager.reset();
  }

  /*! \brief Currently unimplemented interface to specify the active
   *  Hexagon device.
   */
  void SetDevice(Device dev) final {};

  //! \brief Return the queried Hexagon device attribute.
  ffi::Any GetAttr(Device dev, const ffi::String& name) final;

  ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                    ffi::Optional<ffi::String> mem_scope = std::nullopt) final;

  //! \brief Currently unimplemented interface to synchronize a device stream.
  void StreamSync(Device dev, TVMStreamHandle stream) final {}

  //! \note Backend-local allocation helpers for generated Hexagon operators.
  //! \brief Allocate a flat allocation of global memory wrapped in a HexagonBuffer.
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint);

  //! \brief Free the allocated HexagonBuffer.
  void FreeDataSpace(Device dev, void* ptr);

  /*!
   * \brief Allocate an Nd data space on device with memory scope support.
   *
   * If mem_scope is undefined or is "global", treat shape as the
   * tensor shape, to be flattened into an allocation of 1-d physical
   * memory.  This is done to maintain the semantics expected by callers of
   * compact tensor allocation, in cases where it has a valid return value.
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
                       ffi::Optional<ffi::String> mem_scope);

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
    TVM_FFI_ICHECK(runtime_threads) << "runtime_threads has not been created";
    return runtime_threads.get();
  }

  HexagonUserDMA* UserDMA() {
    TVM_FFI_ICHECK(runtime_dma) << "runtime_dma has not been created";
    return runtime_dma.get();
  }

  // The alias also retains power for allocations that outlive their session.
  std::shared_ptr<HexagonVtcmPool> VtcmPoolOwner() {
    TVM_FFI_ICHECK(runtime_vtcm) << "runtime_vtcm has not been created";
    return runtime_vtcm;
  }

  HexagonVtcmPool* VtcmPool() {
    TVM_FFI_ICHECK(runtime_vtcm) << "runtime_vtcm has not been created";
    return runtime_vtcm.get();
  }

 protected:
  //! Standard Device API interface to copy data from one storage to another.
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final;

 private:
  // VTCM reserves the device's available memory. Surviving VTCM allocations and
  // a new session must share that reservation instead of requesting it twice.
  // Member order keeps power available until after the VTCM reservation is freed.
  struct DeviceResources {
    HexagonPowerManager power;
    HexagonVtcmPool vtcm;
  };

  static std::shared_ptr<DeviceResources> AcquireDeviceResources() {
    struct ResourceCache {
      std::recursive_mutex mutex;
      std::condition_variable_any released;
      std::weak_ptr<DeviceResources> resources;
      bool reserved{false};
    };
    // The cache also outlives tensor deleters during process shutdown.
    static auto* cache = new ResourceCache();
    std::unique_lock<std::recursive_mutex> lock(cache->mutex);
    while (true) {
      if (auto resources = cache->resources.lock()) {
        return resources;
      }
      // A weak lock can fail before the final deleter releases the reservation.
      // Let that deleter finish before asking HAP for the device memory again.
      if (cache->reserved) {
        cache->released.wait(lock);
        continue;
      }
      auto* resource = new DeviceResources();
      cache->reserved = true;
      auto resources =
          std::shared_ptr<DeviceResources>(resource, [state = cache](DeviceResources* p) {
            // Recursive locking also handles a failed shared_ptr control-block allocation.
            std::lock_guard<std::recursive_mutex> lock(state->mutex);
            delete p;
            state->reserved = false;
            state->released.notify_all();
          });
      cache->resources = resources;
      return resources;
    }
  }

  /*! \brief Helper to check if the device type is valid for the Hexagon Device API
   *  \return Boolean indicating whether the device type is valid
   */
  bool IsValidDevice(DLDevice dev) {
    // Added kDLCPU since we use hexagon as a sub-target of LLVM which by default maps to kDLCPU
    return (dev.device_type == kDLHexagon) || (dev.device_type == kDLCPU);
  }

  //! \brief Manages runtime HexagonBuffer allocations
  // runtime_hexbuffs is used for runtime allocations.  It is created with a call to
  // AcquireResources. Owning tensors retain their original manager after ReleaseResources,
  // so a late deleter never frees an allocation through a replacement session.
  std::shared_ptr<HexagonBufferManager> runtime_hexbuffs;

  //! \brief Thread manager
  std::unique_ptr<HexagonThreadManager> runtime_threads;
  const unsigned threads{6};
  const unsigned pipe_size{1000};
  const unsigned stack_size{0x4000};  // 16KB
  const std::vector<HardwareResourceType> hw_resources{DMA_0, HTP_0, HVX_0, HVX_1, HVX_2, HVX_3};

  //! \brief User DMA manager
  std::unique_ptr<HexagonUserDMA> runtime_dma;

  //! \brief VTCM memory manager
  std::shared_ptr<HexagonVtcmPool> runtime_vtcm;

  //! \brief Hexagon power manager
  std::shared_ptr<HexagonPowerManager> runtime_power_manager;
};
}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_DEVICE_API_H_
