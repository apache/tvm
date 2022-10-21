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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_VTCM_POOL_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_VTCM_POOL_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace hexagon {

class HexagonVtcmPool {
 public:
  //! \brief Allocates all of VTCM memory, and manages allocations from the runtime
  HexagonVtcmPool();

  //! \brief Destruction deallocates the underlying VTCM allocation.
  ~HexagonVtcmPool();

  //! \brief Prevent copy construction of HexagonVtcmPool.
  HexagonVtcmPool(const HexagonVtcmPool&) = delete;

  //! \brief Prevent copy assignment with HexagonVtcmPool.
  HexagonVtcmPool& operator=(const HexagonVtcmPool&) = delete;

  //! \brief Prevent move construction.
  HexagonVtcmPool(HexagonVtcmPool&&) = delete;

  //! \brief Prevent move assignment.
  HexagonVtcmPool& operator=(HexagonVtcmPool&&) = delete;

  /* \brief Allocate memory from the VTCM manager
   *
   * \param nbytes The number of bytes to allocate.
   */
  void* Allocate(size_t nbytes);

  /* \brief Copy data from a Hexagon Buffer an external buffer.
   *
   * \param ptr The pointer to the buffer to be freed.
   *
   * \param nbytes The number of bytes to be freed.
   */
  void Free(void* ptr, size_t nbytes);

  //! \brief Returns the total number of bytes in this pool
  size_t TotalBytes() { return reinterpret_cast<size_t>(vtcm_size_); }

 private:
  //! \brief Total size of VTCM pool
  unsigned int vtcm_size_;

  //! \brief Pointer to the beginning of the pool
  void* vtcm_data_;

  //! \brief Context for HAP_compute_res_*
  unsigned int context_id_{0};

  //! \brief List of allocations
  std::vector<std::pair<char*, size_t>> allocations_;

  //! \brief List of free segments
  std::vector<std::pair<char*, size_t>> free_;

  //! \brief Mutext to protect access to the lists
  std::mutex mutex_;

  //! \brief Debug only dump of the state of the lists
  void DebugDump();
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_VTCM_POOL_H_
