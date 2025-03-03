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
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

namespace vllm {

template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,      // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,    // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key_cache,      // [num_blocks, num_heads, head_size/x, block_size, x]
    scalar_t* __restrict__ value_cache,    // [num_blocks, num_heads, head_size, block_size]
    const int* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads, const int head_size,
    const int block_size, const int x) {
  const int token_idx = blockIdx.x;
  const int slot_idx = slot_mapping[token_idx];
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int src_key_idx = token_idx * key_stride + i;
    const int src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x +
                            head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
                            block_offset * x + x_offset;
    const int tgt_value_idx = block_idx * num_heads * head_size * block_size +
                              head_idx * head_size * block_size + head_offset * block_size +
                              block_offset;
    key_cache[tgt_key_idx] = __ldg(&key[src_key_idx]);
    value_cache[tgt_value_idx] = __ldg(&value[src_value_idx]);
  }
}

template <typename scalar_t>
__global__ void reconstruct_from_cache_kernel(
    const scalar_t* __restrict__ key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ value_cache,  // [num_blocks, num_heads, head_size, block_size]
    const int* __restrict__ slot_mapping,      // [num_tokens]
    scalar_t* __restrict__ key,                // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ value,              // [num_tokens, num_heads, head_size]
    const int key_stride, const int value_stride, const int num_heads, const int head_size,
    const int block_size, const int x) {
  const int token_idx = blockIdx.x;
  const int slot_idx = slot_mapping[token_idx];
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int tgt_key_idx = token_idx * key_stride + i;
    const int tgt_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int src_key_idx = block_idx * num_heads * (head_size / x) * block_size * x +
                            head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
                            block_offset * x + x_offset;
    const int src_value_idx = block_idx * num_heads * head_size * block_size +
                              head_idx * head_size * block_size + head_offset * block_size +
                              block_offset;

    key[tgt_key_idx] = __ldg(&key_cache[src_key_idx]);
    value[tgt_value_idx] = __ldg(&value_cache[src_value_idx]);
  }
}

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_kernel(int64_t* key_cache_ptrs, int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

}  // namespace vllm

namespace tvm {
namespace runtime {

TVM_REGISTER_GLOBAL("tvm.contrib.vllm.reshape_and_cache")
    .set_body_typed([](NDArray key, NDArray value, NDArray key_cache, NDArray value_cache,
                       NDArray slot_mapping) {
      int num_tokens = key->shape[0];
      int num_heads = key->shape[1];
      int head_size = key->shape[2];
      int block_size = key_cache->shape[3];
      int vec_size = key_cache->shape[4];

      int key_stride = key->shape[1] * key->shape[2];
      int value_stride = value->shape[1] * value->shape[2];

      dim3 grid(num_tokens);
      dim3 block(std::min(num_heads * head_size, 512));

      using scalar_t = uint16_t;
      vllm::reshape_and_cache_kernel<scalar_t><<<grid, block>>>(
          static_cast<const scalar_t*>(key->data), static_cast<const scalar_t*>(value->data),
          static_cast<scalar_t*>(key_cache->data), static_cast<scalar_t*>(value_cache->data),
          static_cast<const int*>(slot_mapping->data), key_stride, value_stride, num_heads,
          head_size, block_size, vec_size);

      return Array{key_cache, value_cache};
    });

TVM_REGISTER_GLOBAL("tvm.contrib.vllm.reconstruct_from_cache")
    .set_body_typed([](NDArray key_cache, NDArray value_cache, NDArray slot_mapping) {
      int num_tokens = slot_mapping->shape[0];
      int num_heads = value_cache->shape[1];
      int head_size = value_cache->shape[2];
      int block_size = value_cache->shape[3];
      int vec_size = key_cache->shape[4];

      DLDevice dev = key_cache->device;
      auto key = NDArray::Empty({num_tokens, num_heads, head_size}, key_cache->dtype, dev);
      auto value = NDArray::Empty({num_tokens, num_heads, head_size}, key_cache->dtype, dev);

      int key_stride = key->shape[1] * key->shape[2];
      int value_stride = value->shape[1] * value->shape[2];

      dim3 grid(num_tokens);
      dim3 block(std::min(num_heads * head_size, 512));

      using scalar_t = uint16_t;
      vllm::reconstruct_from_cache_kernel<scalar_t>
          <<<grid, block>>>(static_cast<const scalar_t*>(key_cache->data),
                            static_cast<const scalar_t*>(value_cache->data),
                            static_cast<const int*>(slot_mapping->data),
                            static_cast<scalar_t*>(key->data), static_cast<scalar_t*>(value->data),
                            key_stride, value_stride, num_heads, head_size, block_size, vec_size);

      return Array{key, value};
    });

TVM_REGISTER_GLOBAL("tvm.contrib.vllm.copy_blocks")
    .set_body_typed([](Array<NDArray> key_value_caches, NDArray block_mapping) {
      auto num_layers = key_value_caches.size() / 2;
      auto num_pairs = block_mapping->shape[0] / 2;

      if (num_layers == 0) {
        return;
      }

      std::vector<int64_t> key_cache_ptrs(num_layers);
      std::vector<int64_t> value_cache_ptrs(num_layers);
      for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        key_cache_ptrs[layer_idx] =
            reinterpret_cast<int64_t>(key_value_caches[2 * layer_idx]->data);
        value_cache_ptrs[layer_idx] =
            reinterpret_cast<int64_t>(key_value_caches[2 * layer_idx + 1]->data);
      }

      NDArray key_cache = key_value_caches[1];  // [num_blocks, num_heads, head_size, block_size]
      DLDevice dev = key_cache->device;

      NDArray key_cache_ptrs_gpu =
          NDArray::Empty({static_cast<int>(num_layers)}, runtime::DataType::Int(64), dev);
      NDArray value_cache_ptrs_gpu =
          NDArray::Empty({static_cast<int>(num_layers)}, runtime::DataType::Int(64), dev);
      key_cache_ptrs_gpu.CopyFromBytes(key_cache_ptrs.data(),
                                       sizeof(int64_t) * key_cache_ptrs.size());
      value_cache_ptrs_gpu.CopyFromBytes(value_cache_ptrs.data(),
                                         sizeof(int64_t) * value_cache_ptrs.size());

      NDArray block_mapping_gpu =
          NDArray::Empty(block_mapping.Shape(), runtime::DataType::Int(64), dev);
      block_mapping_gpu.CopyFromBytes(block_mapping->data,
                                      sizeof(int64_t) * block_mapping->shape[0]);

      const int numel_per_block = key_cache->shape[1] * key_cache->shape[2] * key_cache->shape[3];
      dim3 grid(num_layers, num_pairs);
      dim3 block(std::min(1024, numel_per_block));

      using scalar_t = uint16_t;
      vllm::copy_blocks_kernel<scalar_t>
          <<<grid, block>>>(static_cast<int64_t*>(key_cache_ptrs_gpu->data),
                            static_cast<int64_t*>(value_cache_ptrs_gpu->data),
                            static_cast<int64_t*>(block_mapping_gpu->data), numel_per_block);
    });

}  // namespace runtime
}  // namespace tvm
