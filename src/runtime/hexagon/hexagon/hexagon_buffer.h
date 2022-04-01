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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_HEXAGON_BUFFER_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_HEXAGON_BUFFER_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <vector>

namespace tvm {
namespace runtime {
namespace hexagon {

struct Allocation;

class HexagonBuffer {
 public:
  /* \brief Allocate 1d (contiguous) memory within Hexagon accessible
   * memory scopes.
   *
   * \param nbytes The number of bytes of physical storage
   * to allocate.
   *
   * \param alignment The byte alignment to be used when allocating.
   *
   * \param scope Optional storage scope indicating the memory
   * space in which to allocate. Defaults to global system
   * memory (DDR).
   */
  HexagonBuffer(size_t nbytes, size_t alignment, Optional<String> scope);

  /* \brief Allocate 2d (discontiguous) memory within Hexagon accessible
   * memory scopes.
   *
   * \param nallocs The number of allocations.
   *
   * \param nbytes The number of bytes of physical storage
   * to allocate per allocation.
   *
   * \param alignment The byte alignment to be used when allocating.
   *
   * \param scope Optional storage scope indicating the memory
   * space in which to allocate. Defaults to global system
   * memory (DDR).
   */
  HexagonBuffer(size_t nallocs, size_t nbytes, size_t alignment, Optional<String> scope);

  /* \brief Construct a Hexagon Buffer from an external buffer.
   *
   * \param data The pointer to the external buffer.
   *
   * \param nbytes The size of the external buffer in bytes.
   *
   * \param scope Optional storage scope indicating the memory
   * space in which to allocate. Defaults to global system
   * memory (DDR).
   */
  explicit HexagonBuffer(void* data, size_t nbytes, Optional<String> scope);

  //! \brief Destruction deallocates the underlying allocations.
  ~HexagonBuffer();

  //! \brief Prevent copy construction of HexagonBuffers.
  HexagonBuffer(const HexagonBuffer&) = delete;

  //! \brief Prevent copy assignment with HexagonBuffers.
  HexagonBuffer& operator=(const HexagonBuffer&) = delete;

  //! \brief Prevent move construction.
  HexagonBuffer(HexagonBuffer&&) = delete;

  //! \brief Prevent move assignment.
  HexagonBuffer& operator=(HexagonBuffer&&) = delete;

  /*! \brief Return data pointer
   *
   * The return type depends on the buffer being
   */
  void* GetPointer();

  //! \brief Memory scopes managed by a Hexagon Buffer.
  enum class StorageScope {
    //! \brief System DDR corresponding to global storage.
    kDDR,
    /*! \brief Vector tightly coupled memory corresponding to
     *  global.vtcm storage.
     */
    kVTCM,
  };

  //! \brief Return storage scope of underlying allocation.
  StorageScope GetStorageScope() const;

  /* \brief Copy data from a Hexagon Buffer an external buffer.
   *
   * \param data The pointer to the external buffer.
   *
   * \param nbytes The number of bytes to copy.
   */
  void CopyTo(void* data, size_t nbytes) const;

  /* \brief Copy data from an external buffer to a Hexagon Buffer.
   *
   * \param data The pointer to the external buffer.
   *
   * \param nbytes The number of bytes to copy.
   */
  void CopyFrom(void* data, size_t nbytes);

  /* \brief Copy data from one Hexagon Buffer to another.
   *
   * \param other The other Hexagon Buffer.
   *
   * \param nbytes The number of bytes to copy.
   */
  void CopyFrom(const HexagonBuffer& other, size_t nbytes);

 private:
  //! \brief Return the total number of bytes in this buffer
  size_t TotalBytes() const { return nbytes_per_allocation_ * allocations_.size(); }

  //! \brief Assign a storage scope to the buffer.
  void SetStorageScope(Optional<String> scope);
  /*! \brief Array of raw pointer allocations required by the buffer.
   *
   *  For 1d (contiguous) storage a single allocation will result.
   *  For 2d (discontiguous) storage `nallocs` allocations will result.
   */
  std::vector<void*> allocations_;
  /*! \brief Managed allocations which follow RAII and are released
   *  during destruction.
   */
  std::vector<std::unique_ptr<Allocation>> managed_allocations_;
  /*! \brief The underlying storage type in which the allocation
   *  resides.
   */
  size_t ndim_;
  size_t nbytes_per_allocation_;
  StorageScope storage_scope_;
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_HEXAGON_BUFFER_H_
