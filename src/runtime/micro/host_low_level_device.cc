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
 * \file host_low_level_device.cc
 * \brief emulated low-level micro device implementation on host machine
 */

#include <sys/mman.h>
#include <cstring>
#include <memory>
#include "micro_common.h"
#include "low_level_device.h"

namespace tvm {
namespace runtime {

/*!
 * \brief emulated low-level device on host machine
 */
class HostLowLevelDevice final : public LowLevelDevice {
 public:
  /*!
   * \brief constructor to initialize on-host memory region to act as device
   * \param num_bytes size of the emulated on-device memory region
   */
  explicit HostLowLevelDevice(size_t num_bytes) : size_(num_bytes) {
    size_t size_in_pages = (num_bytes + kPageSize - 1) / kPageSize;
    // TODO(weberlo): Set permissions per section (e.g., read-write perms for
    // the heap, execute perms for text, etc.).
    int mmap_prot = PROT_READ | PROT_WRITE | PROT_EXEC;
    int mmap_flags = MAP_ANONYMOUS | MAP_PRIVATE;
    base_addr_ = reinterpret_cast<std::uintptr_t>(
        mmap(nullptr, size_in_pages * kPageSize, mmap_prot, mmap_flags, -1, 0));
  }

  /*!
   * \brief destructor to deallocate on-host device region
   */
  virtual ~HostLowLevelDevice() {
    munmap(reinterpret_cast<void*>(base_addr_), size_);
  }

  void Read(DevBaseOffset offset, void* buf, size_t num_bytes) {
    void* addr = ToDevPtr(offset).cast_to<void*>();
    std::memcpy(buf, addr, num_bytes);
  }

  void Write(DevBaseOffset offset, const void* buf, size_t num_bytes) {
    void* addr = ToDevPtr(offset).cast_to<void*>();
    std::memcpy(addr, buf, num_bytes);
  }

  void Execute(DevBaseOffset func_offset, DevBaseOffset breakpoint) {
    DevPtr func_addr = ToDevPtr(func_offset);
    reinterpret_cast<void (*)(void)>(func_addr.value())();
  }

  std::uintptr_t base_addr() const final {
    return base_addr_;
  }

  const char* device_type() const final {
    return "host";
  }

 private:
  /*! \brief base address of the micro device memory region */
  std::uintptr_t base_addr_;
  /*! \brief size of memory region */
  size_t size_;
};

const std::shared_ptr<LowLevelDevice> HostLowLevelDeviceCreate(size_t num_bytes) {
  std::shared_ptr<LowLevelDevice> lld =
      std::make_shared<HostLowLevelDevice>(num_bytes);
  return lld;
}

}  // namespace runtime
}  // namespace tvm
