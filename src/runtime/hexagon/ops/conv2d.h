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

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>

#include <algorithm>
#include <cassert>

#ifndef TVM_RUNTIME_HEXAGON_OPS_CONV2D_H_
#define TVM_RUNTIME_HEXAGON_OPS_CONV2D_H_

namespace tvm {
namespace runtime {
namespace hexagon {
namespace conv_utils {
static constexpr auto hexagon_device = DLDevice{static_cast<DLDeviceType>(kDLHexagon), 0};

// Standalone DLTensor: the standalone-ness means that this object owns the shape
// (as opposed to a DLTensor).
template <size_t NDIM>
class SDLTensor : public DLTensor {
 public:
  SDLTensor(void* data_ptr, DLDataType data_type, void* data_space, const int64_t* data_dims)
      : SDLTensor(data_ptr, data_type, data_space) {
    for (size_t i = 0; i < NDIM; ++i) dims[i] = data_dims[i];
  }

  SDLTensor(void* data_ptr, DLDataType data_type, void* data_space,
            std::initializer_list<int64_t> data_dims)
      : SDLTensor(data_ptr, data_type, data_space, data_dims.begin()) {}

  void* GetDataSpace() const { return data_space; }

 private:
  /**
   * @brief Construct SDLTensor
   *
   * @param data_ptr Either points to the same memory as data_space or an array of pointers to the
   * start of each chunk of weight. Since weights can be of varying sizes, this array could contain
   * the pointer to each chunk of memory
   * @param data_type data type of the elements in Tensor
   * @param data_space is meant to store the pointer returned from AllocDataSpace and can be freed
   * by passing it to FreeDataSpace
   */
  SDLTensor(void* data_ptr, DLDataType data_type, void* data_space) : data_space(data_space) {
    data = data_ptr;
    device = hexagon_device;
    ndim = NDIM;
    dtype = data_type;
    shape = dims;
    strides = nullptr;
    byte_offset = 0;
  }

  void* data_space = nullptr;
  int64_t dims[NDIM];
};

inline void* to_ptr(uintptr_t v) { return reinterpret_cast<void*>(v); }

inline uintptr_t to_uint(void* ptr) { return reinterpret_cast<uintptr_t>(ptr); }

constexpr int yxc_to_sm_16b(int y, int x, int c) {
  // Map y,x,c coordinates within a block to the offset (in 16-bit elements)
  // from the beginning of the block in spatial-major layout.
  // 10-bit spatial mask: yyyxcccccx
  assert(y >= 0 && x >= 0 && c >= 0);
  assert(y < 8 && x < 4 && c < 32);
  return y << 7 | (x & 2) << 5 | c << 1 | (x & 1);
}

constexpr int yxc_to_sm_8b(int y, int x, int c) {
  // Map y,x,c coordinates within a block to the offset (in 8-bit elements)
  // from the beginning of the block in spatial-major layout.
  // 10-bit spatial mask: yyyxxxccccc
  assert(y >= 0 && x >= 0 && c >= 0);
  assert(y < 8 && x < 8 && c < 32);
  return y << 8 | x << 5 | c;
}

constexpr int hwio_to_sm_8b(int width, int y, int x, int i, int o) {
  // Map y,x,i,o coordinates within a chunk (assuming the origin at the
  // top-left spatial corner) to the offset (in 8-bit elements) from the
  // beginning of the chunk in spatial-major layout.
  // Spatial mask: p..piiioooooii, where p..p are position bits.
  assert(width >= 1);
  assert(y >= 0 && x >= 0 && i >= 0 && o >= 0);
  assert(i < 32 && o < 32);
  int p = y * width + (width - 1 - x);
  return p << 10 | (i & 0x1c) << 5 | o << 2 | (i & 3);
}

constexpr int hwio_to_sm_16b(int width, int y, int x, int i, int o) {
  // Map y,x,i,o coordinates within a chunk (assuming the origin at the
  // top-left spatial corner) to the offset (in 16-bit elements) from the
  // beginning of the chunk in spatial-major layout.
  // Spatial mask: p..piiiioooooi, where p..p are position bits.
  assert(width >= 1);
  assert(y >= 0 && x >= 0 && i >= 0 && o >= 0);
  assert(i < 32 && o < 32);
  int p = y * width + (width - 1 - x);
  return p << 10 | (i & 0x1e) << 5 | o << 1 | (i & 1);
}

constexpr int round_up(int v, int p2) { return (v + p2 - 1) & -p2; }

// Returns the block address at the given index
// Assumptions
// - The data type of tensor is fp16
// - There is only one batch, and hence n==0
inline uintptr_t nhwc_at(const DLTensor& a, int n, int y, int x, int c) {
  if (y < 0 || y >= a.shape[1]) return uintptr_t(0);
  auto p = static_cast<uintptr_t*>(a.data);
  assert(n == 0);
  return p[y * a.shape[2] * a.shape[3] + x * a.shape[3] + c];
}

// Returns the address of the chunk stored at given index
// Assumptions
// - The data type of tensor is fp16
inline uintptr_t hwio_at(const DLTensor& f, int y, int x, int i, int o) {
  auto p = static_cast<uintptr_t*>(f.data);
  return p[y * f.shape[1] * f.shape[2] * f.shape[3] + x * f.shape[2] * f.shape[3] + i * f.shape[3] +
           o];
}

/**
 * @brief Function to "blockize" the flat input data
 * The term "blockize" is used to mention that the data is stored in non-contiguous blocks
 *
 * The input is mapped into the below mentioned layout (notation similar to index map used for
 * transform layout):
 *
 * For uint8_t type
 * lambda n, h, w, c: n, h//8, w//8, c//32, AXIS_SEPARATOR, h%8, w%8, c%32
 *
 * For uint16_t type
 * lambda n, h, w, c: n, h//8, w//4, c//32, AXIS_SEPARATOR, h%8, (w%4)//2, c%32, w%2
 *
 * where AXIS_SEPARATOR represents split up in the physical layout
 *
 * @param out Pre-allocated output memory pointer
 * @param inp_flat Flat input data pointer
 * @param height
 * @param width
 * @param depth
 */
template <typename T, int block_height, int block_width, int block_depth>
void blockize_hwc(void* out, void* inp_flat, int height, int width, int depth) {
  int (*index_func)(int, int, int);
  if constexpr (std::is_same_v<T, uint8_t>)
    index_func = yxc_to_sm_8b;
  else if constexpr (std::is_same_v<T, uint16_t>)
    index_func = yxc_to_sm_16b;
  else
    LOG_ERROR << "blockize_hwc is only supported for uint8_t and uint16_t types";

  auto inp_data = static_cast<T*>(inp_flat);
  auto out_data = static_cast<uintptr_t*>(out);
  const int stride_x = depth;
  const int stride_y = stride_x * width;

  for (int cy = 0; cy < height; cy += block_height) {
    for (int cx = 0; cx < width; cx += block_width) {
      for (int cc = 0; cc < depth; cc += block_depth) {
        auto block = reinterpret_cast<T*>(*out_data++);
        int max_y = std::min(block_height, height - cy);
        int max_x = std::min(block_width, width - cx);
        int max_c = std::min(block_depth, depth - cc);
        for (int y = 0; y < max_y; ++y) {
          for (int x = 0; x < max_x; ++x) {
            for (int c = 0; c < max_c; ++c) {
              block[index_func(y, x, c)] =
                  inp_data[(cy + y) * stride_y + (cx + x) * stride_x + (cc + c)];
            }
            for (int c = max_c; c < block_depth; ++c) block[index_func(y, x, c)] = 0;
          }
          for (int x = max_x; x < block_width; ++x) {
            for (int c = 0; c < block_depth; ++c) block[index_func(y, x, c)] = 0;
          }
        }

        for (int y = max_y; y < block_height; ++y)
          for (int x = 0; x < block_width; ++x)
            for (int c = 0; c < block_depth; ++c) block[index_func(y, x, c)] = 0;
      }  // cc
    }    // cx
  }      // cy
}

/**
 * @brief Convert back from non-contguous layout to a flat layout
 *
 * @param out_flat Pre-allocated output memory pointer
 * @param inp Blockized input data pointer
 * @param height
 * @param width
 * @param depth
 */
template <typename T, int block_height, int block_width, int block_depth>
void deblockize_hwc(void* out_flat, void* inp, int height, int width, int depth) {
  int (*index_func)(int, int, int);
  if constexpr (std::is_same_v<T, uint8_t>)
    index_func = yxc_to_sm_8b;
  else if constexpr (std::is_same_v<T, uint16_t>)
    index_func = yxc_to_sm_16b;
  else
    LOG_ERROR << "deblockize_hwc is only supported for uint8_t and uint16_t types";

  uintptr_t* inp_data = static_cast<uintptr_t*>(inp);
  T* out_data = static_cast<T*>(out_flat);
  const int stride_x = depth;
  const int stride_y = stride_x * width;

  for (int cy = 0; cy < height; cy += block_height) {
    for (int cx = 0; cx < width; cx += block_width) {
      for (int cc = 0; cc < depth; cc += block_depth) {
        auto block = reinterpret_cast<T*>(*inp_data);
        int max_y = std::min(block_height, height - cy);
        int max_x = std::min(block_width, width - cx);
        int max_c = std::min(block_depth, depth - cc);
        for (int y = 0; y < max_y; ++y) {
          for (int x = 0; x < max_x; ++x) {
            for (int c = 0; c < max_c; ++c) {
              out_data[(cy + y) * stride_y + (cx + x) * stride_x + (cc + c)] =
                  block[index_func(y, x, c)];
            }
          }
        }

        inp_data++;
      }
    }
  }
}

/**
 * @brief Convert the layout of weights from flat to "chunked". The term chunked is explained below:
 *
 * Weights are packed into the below mentioned layout (notation similar to index map):
 * Since weights cannot be exactly represented into a index map notation, the
 * base split up is mentioned below with a few gotchas
 *
 * lambda h, w, i, o: h//8, w//4, o//32, i//32, h%8, w%4, (i%32)//2, o%32, i%2
 *
 * The gotchas are:
 *  - (w%4) is actually stored in the right to left order, as in 3,2,1,0 instead of 0,1,2,3
 *  - The h%8 and (w%4) dimensions are not padded up, leading to chunks of different sizes
 *    (thereby the name "chunked" instead of packed)
 *  - The thinnest chunk of width is stored first. For example, if a kernel is 5x5, the first
 *    chunk along the width has size 1 (representing index 0) and then next one has size 4
 *    representing indices (1,2,3,4)
 *
 * @param out_ptr Base pointer table to be filled with the list of pointers to the first addresses
 * of the "chunked" weights
 * @param out_ptr_size The number of chunks
 * @param out Pointer to pre-allocated output memory
 * @param inp Pointer to flat input data
 * @param height
 * @param width
 * @param idepth
 * @param odepth
 */
void chunkify_hwio_16b(void** out_ptr, int out_ptr_size, void* out, void* inp, int height,
                       int width, int idepth, int odepth);
void chunkify_hwio_8b(void** out_ptr, int out_ptr_size, void* out, void* inp, int height, int width,
                      int idepth, int odepth);

template <typename T, int block_height, int block_width, int block_depth>
SDLTensor<4> prepare_nhwc(tvm::runtime::DeviceAPI* device_api, const DLTensor* nhwc_flat,
                          bool copy_data) {
  tvm::runtime::String vtcm_scope = "global.vtcm";

  // Allocate blocks for activations. We will use the block pointers
  // directly from the allocated area.
  int n = nhwc_flat->shape[0];
  int h = round_up(nhwc_flat->shape[1], block_height);
  int w = round_up(nhwc_flat->shape[2], block_width);
  int c = round_up(nhwc_flat->shape[3], block_depth);
  int64_t shape_2d[2] = {(n * h * w * c) / (block_height * block_width * block_depth),
                         block_height * block_width * block_depth};
  void* nhwc_vtcm =
      device_api->AllocDataSpace(hexagon_device, 2, shape_2d, nhwc_flat->dtype, vtcm_scope);
  if (copy_data) {
    blockize_hwc<T, block_height, block_width, block_depth>(
        nhwc_vtcm, nhwc_flat->data, nhwc_flat->shape[1], nhwc_flat->shape[2], nhwc_flat->shape[3]);
  }

  return SDLTensor<4>(nhwc_vtcm, nhwc_flat->dtype, nhwc_vtcm,
                      {n, h / block_height, w / block_width, c / block_depth});
}

int calculate_num_weight_chunks(int64_t* shape_hwio, int chunk_height, int chunk_width,
                                int chunk_in_channel, int chunk_out_channel);

SDLTensor<4> prepare_hwio(tvm::runtime::DeviceAPI* device_api, const DLTensor* hwio_flat,
                          int num_chunks, void** ptr_table);

SDLTensor<4> prepare_hwio_8b(tvm::runtime::DeviceAPI* device_api, const DLTensor* hwio_flat,
                             int num_chunks, void** ptr_table, int wgt_zp = 0);

template <size_t N>
void release(tvm::runtime::DeviceAPI* device_api, const SDLTensor<N>& tensor) {
  if (auto* data_space = tensor.GetDataSpace()) {
    device_api->FreeDataSpace(hexagon_device, data_space);
  }
}

}  // namespace conv_utils
}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_OPS_CONV2D_H_
