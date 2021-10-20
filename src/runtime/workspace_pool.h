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
 * \file workspace_pool.h
 * \brief Workspace pool utility.
 */
#ifndef TVM_RUNTIME_WORKSPACE_POOL_H_
#define TVM_RUNTIME_WORKSPACE_POOL_H_

#include <tvm/runtime/device_api.h>

#include <memory>
#include <vector>

namespace tvm {
namespace runtime {
/*!
 * \brief A workspace pool to manage
 *
 *  \note We have the following assumption about backend temporal
 *   workspace allocation, and will optimize for such assumption,
 *   some of these assumptions can be enforced by the compiler.
 *
 *  - Only a few allocation will happen, and space will be released after use.
 *  - The release order is usually in reverse order of allocate
 *  - Repeative pattern of same allocations over different runs.
 */
class TVM_DLL WorkspacePool {
 public:
  /*!
   * \brief Create pool with specific device type and device.
   * \param device_type The device type.
   * \param device_api The device API.
   */
  WorkspacePool(DLDeviceType device_type, DeviceAPI* device_api);
  /*! \brief destructor */
  ~WorkspacePool();
  /*!
   * \brief Allocate temporal workspace.
   * \param dev The device of allocation.
   * \param size The size to be allocated.
   */
  void* AllocWorkspace(Device dev, size_t size);
  /*!
   * \brief Free temporal workspace in backend execution.
   *
   * \param dev The device of allocation.
   * \param ptr The pointer to be freed.
   */
  void FreeWorkspace(Device dev, void* ptr);

 private:
  class Pool;
  /*! \brief pool of device local array */
  std::vector<Pool*> array_;
  /*! \brief device type this pool support */
  DLDeviceType device_type_;
  /*! \brief The device API */
  DeviceAPI* device_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_WORKSPACE_POOL_H_
