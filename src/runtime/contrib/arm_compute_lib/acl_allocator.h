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
 * \file src/runtime/contrib/arm_compute_lib/acl_allocator.h
 * \brief ACL Allocator implementation that requests memory from TVM.
 */

#ifndef TVM_RUNTIME_CONTRIB_ARM_COMPUTE_LIB_ACL_ALLOCATOR_H_
#define TVM_RUNTIME_CONTRIB_ARM_COMPUTE_LIB_ACL_ALLOCATOR_H_

#include <arm_compute/runtime/IAllocator.h>
#include <arm_compute/runtime/IMemoryRegion.h>
#include <arm_compute/runtime/MemoryRegion.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <memory>

namespace tvm {
namespace runtime {
namespace contrib {

/*!
 * \brief Override ACL memory allocator and replace with TVM workspace based allocation.
 */
class ACLAllocator : public arm_compute::IAllocator {
 public:
  ACLAllocator() = default;

  /*!
   * \brief Allocate bytes to ACL runtime.
   *
   * Specific implementation requests memory from TVM using their device api.
   *
   * \param size Size to allocate.
   * \param alignment Alignment that the returned pointer should comply with.
   * \return A pointer to the allocated memory.
   */
  void* allocate(size_t size, size_t alignment) override;

  /*!
   * \brief Free memory from ACL runtime.
   *
   * \param ptr Pointer to workspace to free.
   */
  void free(void* ptr) override;

  /*!
   * \brief Create self-managed memory region.
   *
   * \param size Size of the memory region.
   * \param alignment Alignment of the memory region.
   * \return The memory region object.
   */
  std::unique_ptr<arm_compute::IMemoryRegion> make_region(size_t size, size_t alignment) override;

 private:
  /*! \brief Always allocate data in the context of the current CPU. */
  const Device device_{kDLCPU, 0};
  /*! \brief Device API which allows requests for memory from TVM. */
  runtime::DeviceAPI* device_api_ = runtime::DeviceAPI::Get(device_);
};

/*!
 * \brief Memory region that can request TVM memory for ACL to use.
 */
class ACLMemoryRegion : public arm_compute::IMemoryRegion {
 public:
  ACLMemoryRegion(size_t size, size_t alignment);
  ACLMemoryRegion(void* ptr, size_t size);

  ~ACLMemoryRegion() override;

  /*! \brief Prevent instances of this class from being copied (As this class contains
   * pointers). */
  ACLMemoryRegion(const ACLMemoryRegion&) = delete;
  /*! \brief Default move constructor. */
  ACLMemoryRegion(ACLMemoryRegion&&) = default;
  /*! \brief Prevent instances of this class from being copied (As this class
   * contains pointers) */
  ACLMemoryRegion& operator=(const ACLMemoryRegion&) = delete;
  /*! Default move assignment operator. */
  ACLMemoryRegion& operator=(ACLMemoryRegion&&) = default;

  void* buffer() override { return this->ptr_; }

  const void* buffer() const override { return this->ptr_; }

  /*!
   * \brief Extract a sub-region from the memory.
   *
   * \warning Ownership is maintained by the parent memory,
   *          while a wrapped raw memory region is returned by this function.
   *          Thus parent memory should not be released before this.
   *
   * \param offset Offset to the region.
   * \param size Size of the region.
   * \return A wrapped memory sub-region with no ownership of the
   * underlying memory.
   */
  std::unique_ptr<arm_compute::IMemoryRegion> extract_subregion(size_t offset,
                                                                size_t size) override;

 private:
  /*! \brief Points to a region of memory allocated by TVM. */
  void* ptr_;
  /*! \brief A subregion doesn't manage TVM memory so we don't need to free it. */
  bool is_subregion_ = false;
  /*! \brief Always allocate data in the context of the current CPU. */
  const Device device_{kDLCPU, 0};
  /*! \brief Device API which allows requests for memory from TVM. */
  runtime::DeviceAPI* device_api_ = runtime::DeviceAPI::Get(device_);
};

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_ARM_COMPUTE_LIB_ACL_ALLOCATOR_H_
