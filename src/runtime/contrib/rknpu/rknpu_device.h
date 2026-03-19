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
 * \file src/runtime/contrib/rknpu/rknpu_device.h
 * \brief DRM device wrapper for the RK3588 NPU.
 *
 * Provides DMA buffer allocation, cache sync, and task submission
 * via the rknpu Linux DRM driver.
 *
 * Ported from rknpu-compiler/npu/runtime/device.py.
 */

#ifndef TVM_RUNTIME_CONTRIB_RKNPU_RKNPU_DEVICE_H_
#define TVM_RUNTIME_CONTRIB_RKNPU_RKNPU_DEVICE_H_

#ifdef TVM_RKNPU_RUNTIME

#include <cstddef>
#include <cstdint>

namespace tvm {
namespace runtime {
namespace contrib {
namespace rknpu {

// ---------------------------------------------------------------------------
// Memory allocation flags
// ---------------------------------------------------------------------------
static constexpr uint32_t kMemNonContiguous = 1 << 0;
static constexpr uint32_t kMemCacheable = 1 << 1;
static constexpr uint32_t kMemKernelMapping = 1 << 3;
static constexpr uint32_t kMemUnknownBit10 = 1 << 10;
static constexpr uint32_t kMemDefault =
    kMemNonContiguous | kMemCacheable | kMemUnknownBit10;
static constexpr uint32_t kMemTask = kMemDefault | kMemKernelMapping;

// ---------------------------------------------------------------------------
// Sync direction flags
// ---------------------------------------------------------------------------
static constexpr uint32_t kSyncToDevice = 1 << 0;
static constexpr uint32_t kSyncFromDevice = 1 << 1;

// ---------------------------------------------------------------------------
// Job submission flags
// ---------------------------------------------------------------------------
static constexpr uint32_t kJobPC = 1 << 0;
static constexpr uint32_t kJobNonblock = 1 << 1;
static constexpr uint32_t kJobPingpong = 1 << 2;
static constexpr uint32_t kJobBlock = 0;

// ---------------------------------------------------------------------------
// Action types
// ---------------------------------------------------------------------------
static constexpr uint32_t kActReset = 6;

// ---------------------------------------------------------------------------
// IOMMU address limit
// ---------------------------------------------------------------------------
static constexpr uint64_t kIOVALimit = 0xFFF3F000;

// ---------------------------------------------------------------------------
// Kernel ioctl structures (must match rknpu DRM driver)
// ---------------------------------------------------------------------------

#pragma pack(push, 1)
struct RknpuTask {
  uint32_t flags;
  uint32_t op_idx;
  uint32_t enable_mask;
  uint32_t int_mask;
  uint32_t int_clear;
  uint32_t int_status;
  uint32_t regcfg_amount;
  uint32_t regcfg_offset;
  uint64_t regcmd_addr;
};
#pragma pack(pop)

struct RknpuSubcoreTask {
  uint32_t task_start;
  uint32_t task_number;
};

struct RknpuSubmit {
  uint32_t flags;
  uint32_t timeout;
  uint32_t task_start;
  uint32_t task_number;
  uint32_t task_counter;
  int32_t priority;
  uint64_t task_obj_addr;
  int32_t iommu_domain_id;
  uint32_t reserved;
  uint64_t task_base_addr;
  int64_t hw_elapse_time;
  uint32_t core_mask;
  int32_t fence_fd;
  RknpuSubcoreTask subcore_task[5];
};

struct RknpuMemCreate {
  uint32_t handle;
  uint32_t flags;
  uint64_t size;
  uint64_t obj_addr;
  uint64_t dma_addr;
  uint64_t sram_size;
  int32_t iommu_domain_id;
  uint32_t core_mask;
};

struct RknpuMemMap {
  uint32_t handle;
  uint32_t reserved;
  uint64_t offset;
};

struct RknpuMemDestroy {
  uint32_t handle;
  uint32_t reserved;
  uint64_t obj_addr;
};

struct RknpuMemSync {
  uint32_t flags;
  uint32_t reserved;
  uint64_t obj_addr;
  uint64_t offset;
  uint64_t size;
};

struct RknpuAction {
  uint32_t flags;
  uint32_t value;
};

struct DrmVersion {
  int version_major;
  int version_minor;
  int version_patchlevel;
  size_t name_len;
  void* name;
  size_t date_len;
  void* date;
  size_t desc_len;
  void* desc;
};

// ---------------------------------------------------------------------------
// DMA buffer
// ---------------------------------------------------------------------------

struct DMABuffer {
  uint32_t handle = 0;
  uint64_t obj_addr = 0;
  uint32_t dma_addr = 0;
  size_t size = 0;
  void* mapped = nullptr;

  bool Valid() const { return handle != 0 && mapped != nullptr; }

  uint8_t* Data() { return static_cast<uint8_t*>(mapped); }
  const uint8_t* Data() const { return static_cast<const uint8_t*>(mapped); }

  template <typename T>
  T* As() {
    return reinterpret_cast<T*>(mapped);
  }
  template <typename T>
  const T* As() const {
    return reinterpret_cast<const T*>(mapped);
  }
};

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

class RKNPUDevice {
 public:
  RKNPUDevice();
  ~RKNPUDevice();

  RKNPUDevice(const RKNPUDevice&) = delete;
  RKNPUDevice& operator=(const RKNPUDevice&) = delete;

  bool IsOpen() const { return fd_ >= 0; }

  /*!
   * \brief Allocate a DMA buffer and mmap it into user space.
   * \param size Buffer size in bytes.
   * \param flags Memory allocation flags (default: kMemDefault).
   * \param core_mask Core affinity mask (default: 0 = any core).
   * \return Allocated DMA buffer.
   */
  DMABuffer Alloc(size_t size, uint32_t flags = kMemDefault, uint32_t core_mask = 0);

  /*!
   * \brief Free a previously allocated DMA buffer.
   */
  void Free(DMABuffer& buf);

  /*!
   * \brief Flush CPU caches so the NPU sees the latest data.
   */
  void SyncToDevice(const DMABuffer& buf, size_t offset = 0, size_t size = 0);

  /*!
   * \brief Invalidate CPU caches so the CPU sees NPU-written data.
   */
  void SyncFromDevice(const DMABuffer& buf, size_t offset = 0, size_t size = 0);

  /*!
   * \brief Submit a task to the NPU.
   * \param task_buf Buffer containing RknpuTask struct(s), allocated with kMemTask.
   * \param core_mask 1=core0, 2=core1, 4=core2, 7=all three.
   * \param num_tasks Number of tasks in the buffer.
   * \param task_start Index of the first task.
   * \param flags Submit flags (-1 for default: PC|BLOCK|PINGPONG).
   * \param timeout Timeout in milliseconds.
   * \return Hardware elapsed time in nanoseconds.
   */
  int64_t Submit(const DMABuffer& task_buf, uint32_t core_mask, uint32_t num_tasks = 1,
                 uint32_t task_start = 0, int flags = -1, uint32_t timeout = 1000);

  /*!
   * \brief Best-effort submit variant that never FATALs on ioctl failure.
   * \return 0 on success, negative errno-style code on failure.
   */
  int TrySubmit(const DMABuffer& task_buf, uint32_t core_mask, uint32_t num_tasks = 1,
                uint32_t task_start = 0, int flags = -1, uint32_t timeout = 1000,
                int64_t* hw_elapse_time = nullptr);

  /*! \brief Reset the NPU. */
  void Reset();

  /*! \brief Close the DRM device. */
  void Close();

 private:
  int fd_ = -1;

  static int FindDevice();
  static int TryOpenRKNPU(const char* path);
  DMABuffer AllocRaw(size_t size, uint32_t flags, uint32_t core_mask);
  void DestroyHandle(uint32_t handle, uint64_t obj_addr);
};

}  // namespace rknpu
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RKNPU_RUNTIME
#endif  // TVM_RUNTIME_CONTRIB_RKNPU_RKNPU_DEVICE_H_
