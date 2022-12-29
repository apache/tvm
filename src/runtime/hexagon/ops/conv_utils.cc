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

#include <type_traits>

#include "conv2d.h"

namespace tvm {
namespace runtime {
namespace hexagon {
namespace conv_utils {

/**
 * @brief Convert the layout of weights from flat to "chunked". The term chunked is explained below:
 *
 * Weights are packed into the below mentioned layout (notation similar to index map):
 * Since weights cannot be exactly represented into a index map notation, the
 * base split up is mentioned below with a few deviations
 *
 * lambda h, w, i, o: o//32, i//32, h, w, (i%32)//4, o%32, i%4
 *
 * The deviations are:
 *  - w is actually stored in the right to left order, as in 3,2,1,0 instead of 0,1,2,3
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
void chunkify_hwio_8b(void** out_ptr, int out_ptr_size, void* out, void* inp, int height, int width,
                      int idepth, int odepth, int wgt_zp) {
  auto inp_data = static_cast<int8_t*>(inp);
  auto out_data = static_cast<int8_t*>(out);
  const int stride_i = odepth;
  const int stride_x = stride_i * idepth;
  const int stride_y = stride_x * width;

  for (int ci = 0; ci < idepth; ci += 32) {
    for (int co = 0; co < odepth; co += 32) {
      int max_i = std::min(32, idepth - ci);
      int max_o = std::min(32, odepth - co);

      auto chunk = out_data;
      for (int y = 0; y < height; ++y) {
        for (int x = width - 1; x >= 0; --x) {
          for (int i = 0; i < max_i; ++i) {
            for (int o = 0; o < max_o; ++o) {
              chunk[hwio_to_sm_8b(width, y, x, i, o)] =
                  inp_data[y * stride_y + x * stride_x + (ci + i) * stride_i + (co + o)];
            }
            for (int o = max_o; o < 32; ++o) chunk[hwio_to_sm_8b(width, y, x, i, o)] = wgt_zp;
          }
          for (int i = max_i; i < 32; ++i)
            for (int o = 0; o < 32; ++o) chunk[hwio_to_sm_8b(width, y, x, i, o)] = wgt_zp;
        }
      }

      *out_ptr++ = chunk;
      out_data += height * width * 32 * 32;
      out_ptr_size--;
      assert(out_ptr_size >= 0);
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
                       int width, int idepth, int odepth) {
  auto inp_data = static_cast<uint16_t*>(inp);
  auto out_data = static_cast<uint16_t*>(out);
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

          auto chunk = out_data;
          for (int y = 0; y < max_y; ++y) {
            for (int x = max_x - 1; x >= 0; --x) {
              for (int i = 0; i < max_i; ++i) {
                for (int o = 0; o < max_o; ++o) {
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

std::tuple<int, int, int, int> getHWIO(const DLTensor* hwio_flat) {
  int h = hwio_flat->shape[0];
  int w = hwio_flat->shape[1];
  int i = round_up(hwio_flat->shape[2], 32);
  int o = round_up(hwio_flat->shape[3], 32);
  return std::make_tuple(h, w, i, o);
}

SDLTensor<4> prepare_hwio_8b(tvm::runtime::DeviceAPI* device_api, const DLTensor* hwio_flat,
                             int num_chunks, void** ptr_table, int wgt_zp) {
  tvm::runtime::String vtcm_scope = "global.vtcm";

  auto [h, w, i, o] = getHWIO(hwio_flat);
  int64_t shape_1d[] = {h * w * i * o};
  void* hwio_vtcm =
      device_api->AllocDataSpace(hexagon_device, 1, shape_1d, hwio_flat->dtype, vtcm_scope);

  chunkify_hwio_8b(ptr_table, num_chunks, hwio_vtcm, hwio_flat->data, hwio_flat->shape[0],
                   hwio_flat->shape[1], hwio_flat->shape[2], hwio_flat->shape[3], wgt_zp);

  return SDLTensor<4>(ptr_table, hwio_flat->dtype, hwio_vtcm, {1, 1, i / 32, o / 32});
}

SDLTensor<4> prepare_hwio(tvm::runtime::DeviceAPI* device_api, const DLTensor* hwio_flat,
                          int num_chunks, void** ptr_table) {
  tvm::runtime::String vtcm_scope = "global.vtcm";

  // Allocate one block for filter data. We will need to create our own
  // pointer table. The reason is that filter chunks cannot be padded
  // height- or width-wise, so filter chunks may have different sizes.
  // A filter chunk is a block of size HxWx32x32, where H, W are at most
  // height and width of a block respectively.
  auto [h, w, i, o] = getHWIO(hwio_flat);
  int64_t shape_1d[] = {h * w * i * o};
  void* hwio_vtcm =
      device_api->AllocDataSpace(hexagon_device, 1, shape_1d, hwio_flat->dtype, vtcm_scope);

  chunkify_hwio_16b(ptr_table, num_chunks, hwio_vtcm, hwio_flat->data, hwio_flat->shape[0],
                    hwio_flat->shape[1], hwio_flat->shape[2], hwio_flat->shape[3]);

  return SDLTensor<4>(ptr_table, hwio_flat->dtype, hwio_vtcm,
                      {round_up(h, 8) / 8, round_up(w, 4) / 4, i / 32, o / 32});
}

int calculate_num_weight_chunks(int64_t* shape_hwio, int chunk_height, int chunk_width,
                                int chunk_in_channel, int chunk_out_channel) {
  // Define slower roundup that doesn't assume multiplier 'p' to be power of 2
  auto roundup = [](int v, int p) { return (v + p - 1) - ((v + p - 1) % p); };
  int h = roundup(shape_hwio[0], chunk_height);
  int w = roundup(shape_hwio[1], chunk_width);
  int i = roundup(shape_hwio[2], chunk_in_channel);
  int o = roundup(shape_hwio[3], chunk_out_channel);

  return (h * w * i * o) / (chunk_height * chunk_width * chunk_in_channel * chunk_out_channel);
}

}  // namespace conv_utils
}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
