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
 * \file texture.h
 * \brief Texture utilities
 */
#ifndef TVM_RUNTIME_TEXTURE_POOL_H_
#define TVM_RUNTIME_TEXTURE_POOL_H_

#include <tvm/runtime/device_api.h>

#include <memory>
#include <vector>

namespace tvm {
namespace runtime {

class TVM_DLL TexturePool {
 public:
  /*!
   * \brief Create pool with specific device type and device.
   * \param device_type The device type.
   * \param device_api The device API.
   */
  TexturePool(DLDeviceType device_type, DeviceAPI* device_api);
  /*! \brief destructor */
  ~TexturePool();
  /*!
   * \brief Allocate temporal texture.
   * \param ctx The context of allocation.
   * \param width The width of the 2d texture to be allocated.
   * \param height The height of the 2d texture to be allocated.
   */
  void* AllocTexture(TVMContext ctx, size_t width, size_t height, DLDataType type_hint);
  /*!
   * \brief Free temporal texture in backend execution.
   *
   * \param ctx The context of allocation.
   * \param ptr The pointer to be freed.
   */
  void FreeTexture(TVMContext ctx, void* ptr);

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
#endif  // TVM_RUNTIME_TEXTURE_POOL_H_
