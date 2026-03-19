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
 * \file src/runtime/contrib/rknpu/rknpu_device.cc
 * \brief DRM device wrapper implementation for the RK3588 NPU.
 *
 * Ported from rknpu-compiler/npu/runtime/device.py.
 */

#ifdef TVM_RKNPU_RUNTIME

#include "rknpu_device.h"

#include <cerrno>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <tvm/runtime/logging.h>
#include <unistd.h>

#include <cstring>

namespace tvm {
namespace runtime {
namespace contrib {
namespace rknpu {

// ---------------------------------------------------------------------------
// DRM ioctl command numbers
// ---------------------------------------------------------------------------

static constexpr uint8_t kDrmBase = 'd';
static constexpr uint8_t kDrmCommandBase = 0x40;

// RKNPU ioctl offsets
static constexpr uint8_t kNrAction = 0x00;
static constexpr uint8_t kNrSubmit = 0x01;
static constexpr uint8_t kNrMemCreate = 0x02;
static constexpr uint8_t kNrMemMap = 0x03;
static constexpr uint8_t kNrMemDestroy = 0x04;
static constexpr uint8_t kNrMemSync = 0x05;

// _IOWR('d', nr, type) = (3 << 30) | (sizeof(type) << 16) | ('d' << 8) | nr
template <typename T>
static constexpr unsigned long IoctlCmd(uint8_t nr) {
  return (3UL << 30) | (static_cast<unsigned long>(sizeof(T)) << 16) |
         (static_cast<unsigned long>(kDrmBase) << 8) | nr;
}

static constexpr auto kCmdVersion = IoctlCmd<DrmVersion>(0x00);
static constexpr auto kCmdAction = IoctlCmd<RknpuAction>(kDrmCommandBase + kNrAction);
static constexpr auto kCmdSubmit = IoctlCmd<RknpuSubmit>(kDrmCommandBase + kNrSubmit);
static constexpr auto kCmdMemCreate = IoctlCmd<RknpuMemCreate>(kDrmCommandBase + kNrMemCreate);
static constexpr auto kCmdMemMap = IoctlCmd<RknpuMemMap>(kDrmCommandBase + kNrMemMap);
static constexpr auto kCmdMemDestroy = IoctlCmd<RknpuMemDestroy>(kDrmCommandBase + kNrMemDestroy);
static constexpr auto kCmdMemSync = IoctlCmd<RknpuMemSync>(kDrmCommandBase + kNrMemSync);

// ---------------------------------------------------------------------------
// Device discovery
// ---------------------------------------------------------------------------

int RKNPUDevice::TryOpenRKNPU(const char* path) {
  int fd = open(path, O_RDWR);
  if (fd < 0) return -1;

  char name_buf[256] = {};
  char date_buf[256] = {};
  char desc_buf[256] = {};

  DrmVersion dv;
  memset(&dv, 0, sizeof(dv));
  dv.name_len = sizeof(name_buf);
  dv.name = name_buf;
  dv.date_len = sizeof(date_buf);
  dv.date = date_buf;
  dv.desc_len = sizeof(desc_buf);
  dv.desc = desc_buf;

  if (ioctl(fd, kCmdVersion, &dv) < 0) {
    close(fd);
    return -1;
  }

  if (strncmp(name_buf, "rknpu", 5) != 0) {
    close(fd);
    return -1;
  }

  return fd;
}

int RKNPUDevice::FindDevice() {
  // Try renderD128 first (most common on RK3588).
  int fd = TryOpenRKNPU("/dev/dri/renderD128");
  if (fd >= 0) return fd;

  // Scan card0..card7.
  char path[64];
  for (int i = 0; i < 8; i++) {
    snprintf(path, sizeof(path), "/dev/dri/card%d", i);
    fd = TryOpenRKNPU(path);
    if (fd >= 0) return fd;
  }

  // Scan renderD129..renderD135.
  for (int i = 129; i < 136; i++) {
    snprintf(path, sizeof(path), "/dev/dri/renderD%d", i);
    fd = TryOpenRKNPU(path);
    if (fd >= 0) return fd;
  }

  return -1;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

RKNPUDevice::RKNPUDevice() {
  fd_ = FindDevice();
  if (fd_ < 0) {
    LOG(FATAL) << "RKNPUDevice: could not find an rknpu DRM device";
  }
}

RKNPUDevice::~RKNPUDevice() { Close(); }

void RKNPUDevice::Close() {
  if (fd_ >= 0) {
    Reset();
    close(fd_);
    fd_ = -1;
  }
}

void RKNPUDevice::Reset() {
  if (fd_ < 0) return;
  RknpuAction act;
  memset(&act, 0, sizeof(act));
  act.flags = kActReset;
  ioctl(fd_, kCmdAction, &act);  // ignore errors
}

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

DMABuffer RKNPUDevice::AllocRaw(size_t size, uint32_t flags, uint32_t core_mask) {
  DMABuffer buf;

  // Step 1: create kernel DMA memory.
  RknpuMemCreate mc;
  memset(&mc, 0, sizeof(mc));
  mc.flags = flags;
  mc.size = size;
  mc.core_mask = core_mask;

  if (ioctl(fd_, kCmdMemCreate, &mc) < 0) {
    LOG(FATAL) << "RKNPUDevice: MEM_CREATE failed (size=" << size << "): " << strerror(errno);
    return buf;
  }

  buf.handle = mc.handle;
  buf.obj_addr = mc.obj_addr;
  buf.dma_addr = static_cast<uint32_t>(mc.dma_addr & 0xFFFFFFFF);
  buf.size = size;

  // Step 2: get mmap offset.
  RknpuMemMap mm;
  memset(&mm, 0, sizeof(mm));
  mm.handle = buf.handle;

  if (ioctl(fd_, kCmdMemMap, &mm) < 0) {
    DestroyHandle(buf.handle, buf.obj_addr);
    LOG(FATAL) << "RKNPUDevice: MEM_MAP failed: " << strerror(errno);
    return buf;
  }

  // Step 3: mmap into user space.
  void* mapped =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, static_cast<off_t>(mm.offset));
  if (mapped == MAP_FAILED) {
    DestroyHandle(buf.handle, buf.obj_addr);
    LOG(FATAL) << "RKNPUDevice: mmap failed: " << strerror(errno);
    return buf;
  }

  buf.mapped = mapped;
  return buf;
}

DMABuffer RKNPUDevice::Alloc(size_t size, uint32_t flags, uint32_t core_mask) {
  DMABuffer buf = AllocRaw(size, flags, core_mask);

  // If the buffer's usable range exceeds the IOMMU limit, pad the
  // allocation so the kernel places it at a lower IOVA base.
  if (static_cast<uint64_t>(buf.dma_addr) + size > kIOVALimit) {
    uint64_t padding = (static_cast<uint64_t>(buf.dma_addr) + size) - kIOVALimit;
    padding = (padding + 0xFFF) & ~0xFFFULL;  // round up to page
    Free(buf);
    buf = AllocRaw(size + padding, flags, core_mask);
  }

  return buf;
}

void RKNPUDevice::Free(DMABuffer& buf) {
  if (buf.mapped != nullptr) {
    munmap(buf.mapped, buf.size);
    buf.mapped = nullptr;
  }
  if (buf.handle != 0) {
    DestroyHandle(buf.handle, buf.obj_addr);
    buf.handle = 0;
  }
}

void RKNPUDevice::DestroyHandle(uint32_t handle, uint64_t obj_addr) {
  RknpuMemDestroy md;
  memset(&md, 0, sizeof(md));
  md.handle = handle;
  md.obj_addr = obj_addr;
  ioctl(fd_, kCmdMemDestroy, &md);  // ignore errors
}

// ---------------------------------------------------------------------------
// Cache sync
// ---------------------------------------------------------------------------

void RKNPUDevice::SyncToDevice(const DMABuffer& buf, size_t offset, size_t size) {
  if (buf.obj_addr == 0) return;
  size_t sync_size = (size > 0) ? size : buf.size;

  RknpuMemSync ms;
  memset(&ms, 0, sizeof(ms));
  ms.flags = kSyncToDevice;
  ms.obj_addr = buf.obj_addr;
  ms.offset = offset;
  ms.size = sync_size;

  if (ioctl(fd_, kCmdMemSync, &ms) < 0) {
    LOG(FATAL) << "RKNPUDevice: SYNC_TO_DEVICE failed: " << strerror(errno);
  }
}

void RKNPUDevice::SyncFromDevice(const DMABuffer& buf, size_t offset, size_t size) {
  if (buf.obj_addr == 0) return;
  size_t sync_size = (size > 0) ? size : buf.size;

  RknpuMemSync ms;
  memset(&ms, 0, sizeof(ms));
  ms.flags = kSyncFromDevice;
  ms.obj_addr = buf.obj_addr;
  ms.offset = offset;
  ms.size = sync_size;

  ioctl(fd_, kCmdMemSync, &ms);  // ignore errors (match C/Python behaviour)
}

// ---------------------------------------------------------------------------
// Task submission
// ---------------------------------------------------------------------------

int RKNPUDevice::TrySubmit(const DMABuffer& task_buf, uint32_t core_mask, uint32_t num_tasks,
                           uint32_t task_start, int flags, uint32_t timeout,
                           int64_t* hw_elapse_time) {
  RknpuSubmit sub;
  memset(&sub, 0, sizeof(sub));

  sub.flags = (flags >= 0) ? static_cast<uint32_t>(flags) : (kJobPC | kJobBlock | kJobPingpong);
  sub.timeout = timeout;
  sub.task_start = task_start;
  sub.task_number = num_tasks;
  sub.task_counter = 0;
  sub.task_obj_addr = task_buf.obj_addr;
  sub.task_base_addr = 0;
  sub.core_mask = core_mask;
  sub.fence_fd = -1;

  // Assign tasks to cores.
  if (core_mask == 1 || core_mask == 2 || core_mask == 4) {
    // Single core.
    int core_idx = (core_mask == 1) ? 0 : (core_mask == 2) ? 1 : 2;
    sub.subcore_task[core_idx].task_start = task_start;
    sub.subcore_task[core_idx].task_number = num_tasks;
  } else {
    // Multi-core: distribute tasks across active cores.
    int active[3];
    int count = 0;
    for (int i = 0; i < 3; i++) {
      if (core_mask & (1 << i)) {
        active[count++] = i;
      }
    }

    // Reduce core_mask if more cores than tasks.
    int cores_needed = (count < static_cast<int>(num_tasks)) ? count : num_tasks;
    if (cores_needed < count) {
      count = cores_needed;
      uint32_t new_mask = 0;
      for (int i = 0; i < count; i++) new_mask |= (1 << active[i]);
      core_mask = new_mask;
      sub.core_mask = core_mask;
    }

    int per_core = num_tasks / count;
    int remainder = num_tasks % count;
    // Kernel quirk: for 3-core mode, driver reads subcore_task[core_index + 2].
    int idx_offset = (count == 3) ? 2 : 0;
    uint32_t offset = task_start;
    for (int i = 0; i < count; i++) {
      int n = per_core + ((i < remainder) ? 1 : 0);
      sub.subcore_task[active[i] + idx_offset].task_start = offset;
      sub.subcore_task[active[i] + idx_offset].task_number = n;
      offset += n;
    }
  }

  if (ioctl(fd_, kCmdSubmit, &sub) < 0) {
    return -errno;
  }

  if (hw_elapse_time != nullptr) {
    *hw_elapse_time = sub.hw_elapse_time;
  }
  return 0;
}

int64_t RKNPUDevice::Submit(const DMABuffer& task_buf, uint32_t core_mask, uint32_t num_tasks,
                            uint32_t task_start, int flags, uint32_t timeout) {
  int64_t hw_time = -1;
  int rc = TrySubmit(task_buf, core_mask, num_tasks, task_start, flags, timeout, &hw_time);
  if (rc < 0) {
    Reset();
    LOG(FATAL) << "RKNPUDevice: SUBMIT failed (core_mask=0x" << std::hex << core_mask
               << "): " << strerror(-rc);
    return -1;
  }
  return hw_time;
}

}  // namespace rknpu
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RKNPU_RUNTIME
