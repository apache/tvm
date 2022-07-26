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

#include <HAP_farf.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>

#include <cassert>

#ifndef TVM_RUNTIME_HEXAGON_OPS_CONV2D_H_
#define TVM_RUNTIME_HEXAGON_OPS_CONV2D_H_

#ifdef DEBUG_CONV
#define DEBUG_BLOCK(X) \
  { X }
#define debug(...) FARF(ALWAYS, ##__VA_ARGS__)
#else
#define DEBUG_BLOCK(X)
#define debug(...)
#endif

#define HAP_CALL(hap_fn, ...)                 \
  {                                           \
    int rc = hap_fn(__VA_ARGS__);             \
    if (rc != 0) {                            \
      debug("%s failed: rc=%x", #hap_fn, rc); \
    }                                         \
  }

namespace detail {
static constexpr auto hexagon_device = DLDevice{static_cast<DLDeviceType>(kDLHexagon), 0};

// Standalone DLTensor: the standalone-ness means that this object owns the shape
// (as opposed to a DLTensor).
template <size_t N>
class SDLTensor : public DLTensor {
 public:
  SDLTensor(void* data_ptr, DLDataType data_type, void* data_space, const int64_t* data_dims)
      : SDLTensor(data_ptr, data_type, data_space) {
    for (size_t i = 0; i != N; ++i) dims[i] = data_dims[i];
  }

  SDLTensor(void* data_ptr, DLDataType data_type, void* data_space,
            std::initializer_list<int64_t> data_dims)
      : SDLTensor(data_ptr, data_type, data_space, data_dims.begin()) {}

  void* GetDataSpace() const { return data_space; }

 private:
  SDLTensor(void* data_ptr, DLDataType data_type, void* data_space) : data_space(data_space) {
    data = data_ptr;
    device = hexagon_device;
    ndim = N;
    dtype = data_type;
    shape = dims;
    strides = nullptr;
    byte_offset = 0;
  }

  void* data_space = nullptr;
  int64_t dims[N];
};

inline void* to_ptr(uintptr_t v) { return reinterpret_cast<void*>(v); }

inline uintptr_t to_uint(void* ptr) { return reinterpret_cast<uintptr_t>(ptr); }

inline constexpr int xyc_to_sm_16b(int y, int x, int c) {
  // Map y,x,c coordinates within a block to the offset (in 16-bit elements)
  // from the beginning of the block in spatial-major layout.
  // 10-bit spatial mask: yyyxcccccx
  return y << 7 | (x & 2) << 5 | c << 1 | (x & 1);
}

inline constexpr int hwio_to_sm_16b(int width, int y, int x, int i, int o) {
  // Map y,x,i,o coordinates within a chunk (assuming the origin at the
  // top-left spatial corner) to the offset (in 16-bit elements) from the
  // beginning of the chunk in spatial-major layout.
  // Spatial mask: p..piiiioooooi, where p..p are position bits.
  int p = y * width + (width - 1 - x);
  return p << 10 | (i & 0x1e) << 5 | o << 1 | (i & 1);
}

inline constexpr int round_up(int v, int p2) { return (v + p2 - 1) & -p2; }

constexpr uintptr_t nhwc_at(const DLTensor& a, int n, int y, int x, int c) {
  if (y < 0 || y >= a.shape[1]) return uintptr_t(0);
  auto p = static_cast<uintptr_t*>(a.data);
  assert(n == 0);
  return p[y * a.shape[2] * a.shape[3] + x * a.shape[3] + c];
}

constexpr uintptr_t hwio_at(const DLTensor& f, int y, int x, int i, int o) {
  auto p = static_cast<uintptr_t*>(f.data);
  return p[y * f.shape[1] * f.shape[2] * f.shape[3] + x * f.shape[2] * f.shape[3] + i * f.shape[3] +
           o];
}

constexpr uint32_t* bias_at(const DLTensor& b, int d) {
  auto p = static_cast<uint32_t*>(b.data);
  return p + d;
}

void blockize_hwc_16b(void* out, void* inp_flat, int height, int width, int depth);

void deblockize_hwc_16b(void* out_flat, void* inp, int height, int width, int depth);

void chunkify_hwio_16b(void** out_ptr, int out_ptr_size, void* out, void* inp, int height,
                       int width, int idepth, int odepth);

SDLTensor<4> prepare_nhwc(tvm::runtime::DeviceAPI* device_api, const DLTensor* nhwc_flat,
                          bool copy_data);

int calculate_num_weight_chunks(int64_t* shape_hwio);

SDLTensor<4> prepare_hwio(tvm::runtime::DeviceAPI* device_api, const DLTensor* hwio_flat,
                          int num_chunks, void** ptr_table);

template <size_t N>
void release(tvm::runtime::DeviceAPI* device_api, const SDLTensor<N>& tensor) {
  if (auto* data_space = tensor.GetDataSpace()) {
    device_api->FreeDataSpace(hexagon_device, data_space);
  }
}

}  // namespace detail

#endif  // TVM_RUNTIME_HEXAGON_OPS_CONV2D_H_
