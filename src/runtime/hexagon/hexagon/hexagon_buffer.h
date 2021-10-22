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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_BUFFER_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_BUFFER_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <vector>

namespace tvm {
namespace runtime {
namespace hexagon {

class HexagonBuffer {
 public:
  /* \brief Allocate memory within hexagon accessible memory
   * scopes.
   *
   * \param ndim The number of dimensions of physical storage
   * to allocate.
   *
   * \param shape The shape of the ndarray for which to allocate
   * physical storage.
   *
   * \param dtype The data type of the physical storage.
   *
   * \param scope Optional storage scope indicating the memory
   * space in which to allocate. Defaults to global system
   * memory (DDR).
   */
  HexagonBuffer(int ndim, const int64_t* shape, DLDataType dtype, Optional<String> scope);

  /* \brief Allocate memory within hexagon accessible memory
   * scopes.
   *
   * \param nbytes The number of bytes of flat physical storage
   * to allocate.
   *
   * \param alignment The byte alignment to be used when allocating.
   *
   * \param scope Optional storage scope indicating the memory
   * space in which to allocate. Defaults to global system
   * memory (DDR).
   */
  HexagonBuffer(size_t nbytes, size_t alignment, Optional<String> scope);

  /* \brief Construct a hexagon buffer from externally allocated storage.
   *
   * \param data The externally allocated storage.
   *
   * \param scope Optional storage scope indicating the memory
   * space in the external allocation belongs. Assumes global system
   * memory if not provided.
   */
  HexagonBuffer(void* data, Optional<String> scope = Optional<String>());

  //! \brief Destruction deallocates the underlying allocations.
  ~HexagonBuffer();

  //! \brief Prevent copy construction of HexagonBuffers.
  HexagonBuffer(const HexagonBuffer&) = delete;

  //! \brief Prevent copy assignment with HexagonBuffers.
  HexagonBuffer& operator=(const HexagonBuffer&) = delete;

  //! \brief Allow move construction.
  HexagonBuffer(HexagonBuffer&&);

  //! \brief Allow move assignment.
  HexagonBuffer& operator=(HexagonBuffer&&);

  //! \brief Return pointer to allocation or allocations.
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

 private:
  //! \brief Assign a storage scope to the buffer.
  void SetStorageScope(Optional<String> scope);
  /*! \brief Array of allocations required by the buffer.
   *
   *  For a 1d (flat) storage, a single contiguous allocation will
   *  result. For 2d storage, (count, nbytes) = shape, which will
   *  result in `count` discrete allocations.
   */
  std::vector<void*> allocations_;
  /*! \brief Whether the allocation(s) present are managed
   *  and should be deallocated upon destruction.
   */
  bool managed_{true};
  /*! \brief The underlying storage type in which the allocation
   *  resides.
   */
  StorageScope storage_scope_;
};

HexagonBuffer* IsHexagonBuffer(DLTensor* tensor);

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_BUFFER_H_
