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

#include "conv2d.h"

namespace detail {

void blockize_hwc_16b(void* out, void* inp_flat, int height, int width, int depth) {
  auto inp_data = static_cast<uint16_t*>(inp_flat);
  auto out_data = static_cast<uintptr_t*>(out);
  const int stride_x = depth;
  const int stride_y = stride_x * width;

  for (int cy = 0; cy < height; cy += 8) {
    for (int cx = 0; cx < width; cx += 4) {
      for (int cc = 0; cc < depth; cc += 32) {
        auto block = reinterpret_cast<uint16_t*>(*out_data++);
        int max_y = std::min(8, height - cy);
        int max_x = std::min(4, width - cx);
        int max_c = std::min(32, depth - cc);
        for (int y = 0; y < max_y; ++y) {
          for (int x = 0; x < max_x; ++x) {
            for (int c = 0; c < max_c; ++c) {
              block[xyc_to_sm_16b(y, x, c)] =
                  inp_data[(cy + y) * stride_y + (cx + x) * stride_x + (cc + c)];
            }
            for (int c = max_c; c < 32; ++c) block[xyc_to_sm_16b(y, x, c)] = 0;
          }
          for (int x = max_x; x < 4; ++x) {
            for (int c = 0; c < 32; ++c) block[xyc_to_sm_16b(y, x, c)] = 0;
          }
        }

        for (int y = max_y; y < 8; ++y)
          for (int x = 0; x < 4; ++x)
            for (int c = 0; c < 32; ++c) block[xyc_to_sm_16b(y, x, c)] = 0;
      }  // cc
    }    // cx
  }      // cy
}

void deblockize_hwc_16b(void* out_flat, void* inp, int height, int width, int depth) {
  uintptr_t* inp_data = static_cast<uintptr_t*>(inp);
  uint16_t* out_data = static_cast<uint16_t*>(out_flat);
  const int stride_x = depth;
  const int stride_y = stride_x * width;

  for (int cy = 0; cy < height; cy += 8) {
    for (int cx = 0; cx < width; cx += 4) {
      for (int cc = 0; cc < depth; cc += 32) {
        auto block = reinterpret_cast<uint16_t*>(*inp_data);
        int max_y = std::min(8, height - cy);
        int max_x = std::min(4, width - cx);
        int max_c = std::min(32, depth - cc);
        for (int y = 0; y < max_y; ++y) {
          for (int x = 0; x < max_x; ++x) {
            for (int c = 0; c < max_c; ++c) {
              out_data[(cy + y) * stride_y + (cx + x) * stride_x + (cc + c)] =
                  block[xyc_to_sm_16b(y, x, c)];
            }
          }
        }

        inp_data++;
      }
    }
  }
}

void chunkify_hwio_16b(void** out_ptr, int out_ptr_size, void* out, void* inp, int height,
                       int width, int idepth, int odepth) {
  auto inp_data = static_cast<uint16_t*>(inp);
  auto out_data = static_cast<uintptr_t*>(out);
  const int stride_i = odepth;
  const int stride_x = stride_i * idepth;
  const int stride_y = stride_x * width;

  for (int cy = 0; cy < height; cy += 8) {
    // In the chunkified tensor, the chunks are ordered in increasing
    // x order, but they start from the thin one.
    for (int cx = width - round_up(width, 4); cx < width; cx += 4) {
      int cx0 = std::max(0, cx);
      for (int ci = 0; ci < idepth; ci += 32) {
        for (int co = 0; co < odepth; co += 32) {
          int max_y = std::min(8, height - cy);
          int max_x = std::min(4, cx + 4 - cx0);
          int max_i = std::min(32, idepth - ci);
          int max_o = std::min(32, odepth - co);

          auto chunk = reinterpret_cast<uint16_t*>(out_data);
          for (int y = 0; y < max_y; ++y) {
            for (int x = max_x - 1; x >= 0; --x) {
              for (int i = 0; i < max_i; ++i) {
                for (int o = 0; o < max_o; ++o) {
                  debug(
                      "cy: %d, cx: %d, cx0: %d, ci: %d, co: %d, max_x: %d, y: %d, x: %d, i: %d, o: "
                      "%d, index: %d",
                      cy, cx, cx0, ci, co, max_x, y, x, i, o, hwio_to_sm_16b(max_x, y, x, i, o));
                  chunk[hwio_to_sm_16b(max_x, y, x, i, o)] =
                      inp_data[(cy + y) * stride_y + (cx0 + x) * stride_x + (ci + i) * stride_i +
                               (co + o)];
                }
                for (int o = max_o; o < 32; ++o) chunk[hwio_to_sm_16b(max_x, y, x, i, o)] = 0;
              }
              for (int i = max_i; i < 32; ++i)
                for (int o = 0; o < 32; ++o) chunk[hwio_to_sm_16b(max_x, y, x, i, o)] = 0;
            }
          }

          *out_ptr++ = chunk;
          out_data += max_y * max_x * 32 * 32;
          out_ptr_size--;
          assert(out_ptr_size >= 0);
        }
      }
    }
  }
}

SDLTensor<4> prepare_nhwc(tvm::runtime::DeviceAPI* device_api, const DLTensor* nhwc_flat,
                          bool copy_data) {
  tvm::runtime::String vtcm_scope = "global.vtcm";

  // Allocate blocks for activations. We will use the block pointers
  // directly from the allocated area.
  int n = nhwc_flat->shape[0];
  int h = round_up(nhwc_flat->shape[1], 8);
  int w = round_up(nhwc_flat->shape[2], 4);
  int c = round_up(nhwc_flat->shape[3], 32);
  int64_t shape_2d[2] = {(n * h * w * c) / (8 * 4 * 32), 8 * 4 * 32};
  void* nhwc_vtcm =
      device_api->AllocDataSpace(hexagon_device, 2, shape_2d, nhwc_flat->dtype, vtcm_scope);
  if (copy_data) {
    blockize_hwc_16b(nhwc_vtcm, nhwc_flat->data, nhwc_flat->shape[1], nhwc_flat->shape[2],
                     nhwc_flat->shape[3]);
  }

  return SDLTensor<4>(nhwc_vtcm, nhwc_flat->dtype, nhwc_vtcm, {n, h / 8, w / 4, c / 32});
}

SDLTensor<4> prepare_hwio(tvm::runtime::DeviceAPI* device_api, const DLTensor* hwio_flat,
                          int num_chunks, void** ptr_table) {
  tvm::runtime::String vtcm_scope = "global.vtcm";

  // Allocate one block for filter data. We will need to create our own
  // pointer table. The reason is that filter chunks cannot be padded
  // height- or width-wise, so filter chunks may have different sizes.
  // A filter chunk is a block of size HxWx32x32, where H, W are at most
  // height and width of a block respectively.
  int h = hwio_flat->shape[0];
  int w = hwio_flat->shape[1];
  int i = round_up(hwio_flat->shape[2], 32);
  int o = round_up(hwio_flat->shape[3], 32);
  int64_t shape_1d[] = {h * w * i * o};
  void* hwio_vtcm =
      device_api->AllocDataSpace(hexagon_device, 1, shape_1d, hwio_flat->dtype, vtcm_scope);

  chunkify_hwio_16b(ptr_table, num_chunks, hwio_vtcm, hwio_flat->data, hwio_flat->shape[0],
                    hwio_flat->shape[1], hwio_flat->shape[2], hwio_flat->shape[3]);

  return SDLTensor<4>(ptr_table, hwio_flat->dtype, hwio_vtcm,
                      {round_up(h, 8) / 8, round_up(w, 4) / 4, i / 32, o / 32});
}

int calculate_num_weight_chunks(int64_t* shape_hwio) {
  int h = round_up(shape_hwio[0], 8);
  int w = round_up(shape_hwio[1], 4);
  int i = round_up(shape_hwio[2], 32);
  int o = round_up(shape_hwio[3], 32);

  debug("h: %d, w: %d, i: %d, o: %d", h, w, i, o);
  return (h * w * i * o) / (8 * 4 * 32 * 32);
}

}  // namespace detail
