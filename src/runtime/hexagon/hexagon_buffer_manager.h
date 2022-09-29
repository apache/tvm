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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_BUFFER_MANAGER_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_BUFFER_MANAGER_H_

#include <memory>
#include <unordered_map>
#include <utility>

#include "hexagon_buffer.h"
#include "hexagon_common.h"

namespace tvm {
namespace runtime {
namespace hexagon {

class HexagonBufferManager {
 public:
  /*!
   * \brief Free a HexagonBuffer.
   * \param ptr Address of the HexagonBuffer as returned by `AllocateHexagonBuffer`.
   */
  void FreeHexagonBuffer(void* ptr) {
    if (auto it = hexagon_buffer_map_.find(ptr); it == hexagon_buffer_map_.end()) {
      // This should be an assertion, but something seems to go wrong here.
      // The symptom is that when resources are being released (ReleaseResources),
      // one buffer disappears from the "runtime" buffer manager, and is not deleted
      // when that manager is reset. The FreeHexagonBuffer is than called for that
      // buffer, but with the "static" buffer manager instead. That manager doesn't
      // find it in the map and throws an exception, which somehow doesn't abort
      // the program.
      HEXAGON_PRINT(ERROR, "Attempt made to free unknown or already freed dataspace allocation");
    } else {
      HEXAGON_ASSERT(it->second != nullptr);
      std::lock_guard<std::mutex> lock(map_mutex_);
      hexagon_buffer_map_.erase(it);
    }
  }
  /*!
   * \brief Allocate a HexagonBuffer.
   * \param args Templated arguments to pass through to HexagonBuffer constructor.
   */
  template <typename... Args>
  void* AllocateHexagonBuffer(Args&&... args) {
    auto buf = std::make_unique<HexagonBuffer>(std::forward<Args>(args)...);
    void* ptr = buf->GetPointer();
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      hexagon_buffer_map_.insert({ptr, std::move(buf)});
    }
    return ptr;
  }

  //! \brief Returns whether the HexagonBuffer is in the map.
  size_t count(void* ptr) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    return hexagon_buffer_map_.count(ptr);
  }

  //! \brief Returns an iterator to the HexagonBuffer within the map.
  HexagonBuffer* find(void* ptr) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto it = hexagon_buffer_map_.find(ptr);
    if (it != hexagon_buffer_map_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  //! \brief Returns whether the HexagonBufferManager has any allocations.
  bool empty() {
    std::lock_guard<std::mutex> lock(map_mutex_);
    return hexagon_buffer_map_.empty();
  }

 private:
  //! \brief Contains the HexagonBuffer objects managed by this class.
  std::unordered_map<void*, std::unique_ptr<HexagonBuffer>> hexagon_buffer_map_;

  //! \brief Protects updates to the map.
  std::mutex map_mutex_;
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_BUFFER_MANAGER_H_
