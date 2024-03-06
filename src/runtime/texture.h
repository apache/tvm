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

#define ALIGN_UP(num, align) (((num) + ((align)-1)) & ~((align)-1))

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

/*!
 * \brief Returns the physical backing memory size required for given specification
 * \param shape shape of tensor
 * \param bits dtype bits
 * \param lanes vectorization lanes
 * \param mem_scope the memory scope info
 * \param image_row_align image rowwise alignment size
 * \return returns the backing memory size
 */
template <typename T>
size_t GetTextureMemorySize(T shape, int bits, int lanes, std::string mem_scope,
                            int image_row_align) {
  size_t axis = DefaultTextureLayoutSeparator(shape.size(), mem_scope);
  auto tshape = ApplyTexture2DFlattening<int64_t>(shape, shape.size(), axis);

  auto pack_size = shape[shape.size() - 1];
  auto pixel_size = (bits * lanes + 7) / 8;
  size_t row_pitch = ALIGN_UP(tshape.width * pixel_size * pack_size, image_row_align);
  return row_pitch * tshape.height;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_TEXTURE_H_
