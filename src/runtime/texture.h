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
#ifndef TVM_RUNTIME_TEXTURE_H_
#define TVM_RUNTIME_TEXTURE_H_

#include <tvm/runtime/device_api.h>

#include <memory>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*! \brief Structure to represent flattened texture shape */
template <typename T>
struct Texture2DShape {
  T width;
  T height;
  T channel;
};

/*!
 * \param shape_rank Rank N of the Nd-shape
 * \param convention Storage scope convention to use for flattening
 * \return The axis separator that defines the Nd shape partitioning in 2d
 */
inline size_t DefaultTextureLayoutSeparator(size_t shape_rank,
                                            std::string convention = "global.texture") {
  // Texture activation:
  // e.g. [N,C,H,W,c] -> Texture2d[N*C*H, W, c]
  // Texture weight:
  // e.g. [O,I,H,W,c] -> Texture2d[O, I*H*W, c]
  size_t separator = 0;
  if (convention == "global.texture") {
    separator = shape_rank - 2;
  } else if (convention == "global.texture-weight") {
    separator = 1;
  } else if (convention == "global.texture-nhwc") {
    if (shape_rank == 3) {
      separator = 1;
    } else {
      separator = 2;
    }
  } else {
    LOG(FATAL) << "Encountered unknown texture lowering convention: " << convention;
  }
  return separator;
}

/*!
 * \param shape Nd shape
 * \param rank Number of dimensions N of the Nd shape
 * \param axis The axis separator that splits the Nd axes into two sets
 * \return Width and height of the 2d shape
 */
template <typename T, typename S>
Texture2DShape<T> ApplyTexture2DFlattening(const S& shape, size_t rank, size_t axis) {
  ICHECK(axis < rank)
      << "Number of axes to flatten into rows must be less than shape rank for 2d flattening";
  Texture2DShape<T> texture{1, 1, shape[rank - 1]};
  for (size_t i = 0; i < rank - 1; i++) {
    if (i < axis) {
      texture.height *= shape[i];
    } else {
      texture.width *= shape[i];
    }
  }
  return texture;
}

inline bool IsTextureStorage(std::string scope) {
  return scope.find("texture") != std::string::npos;
}

class TVM_DLL Pool2D {
 public:
  Pool2D() = default;
  void* Alloc(Device dev, DeviceAPI* device, size_t width, size_t height, DLDataType type_hint);
  void Free(void* data);
  // Release all resources immediately
  void Release(Device dev, DeviceAPI* device);

 protected:
  struct Entry {
    void* data;
    size_t x;
    size_t y;
    DLDataType type;
  };
  std::vector<Entry> free_list_;
  std::vector<Entry> allocated_;
};

/*!
 * \brief A two dimensional storage pool that recycles temporal workspace
 * allocations for dynamically allocated texture. See AllocTexture docstring
 * for approach to allocation and reuse.
 */
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
   * \brief Allocate a two dimensional temporal texture workspace on device
   *
   * \note Two dimensional texture workspaces will be grown and reused
   * according to the following strategy:
   *  - Choose the workspace which minimizes the amount of memory required to
   *    grow the workspace to fit the request.
   *  - If a set of workspaces exist that fit the current request without
   *    expansion, choose the workspace of that set which most closely
   *    matches the request size, minimizing wasted space.
   *
   * \param dev The context of allocation.
   * \param width The width of the 2d texture to be allocated.
   * \param height The height of the 2d texture to be allocated.
   * \param type_hint The type of elements.
   */
  void* AllocTexture(Device dev, size_t width, size_t height, DLDataType type_hint);
  /*!
   * \brief Free temporal texture in backend execution.
   *
   * \param dev The context of allocation.
   * \param ptr The pointer to be freed.
   */
  void FreeTexture(Device dev, void* ptr);

 private:
  /*! \brief pool of device local array */
  std::vector<Pool2D*> array_;
  /*! \brief device type this pool support */
  DLDeviceType device_type_;
  /*! \brief The device API */
  DeviceAPI* device_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_TEXTURE_H_
