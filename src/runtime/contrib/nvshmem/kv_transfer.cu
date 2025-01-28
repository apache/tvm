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
#include <cuda_fp16.h>
#include <dlpack/dlpack.h>
#include <nvshmem.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

template <int dim>
__device__ int64_t calc_flattened_index(int shape[dim], int index[dim]) {
  int64_t flattened_index = 0;
#pragma unroll
  for (int i = 0; i < dim; i++) {
    flattened_index *= shape[i];
    flattened_index += index[i];
  }
  return flattened_index;
}

template <typename T, int local_num_kv_head, int remote_num_kv_head, int head_dim, int page_size>
__global__ void KVTransfer(T* pages, T* k_data, T* v_data, int32_t* remote_position_map,
                           int ntokens, int local_tp_rank, int32_t* remote_tp_group_pe_offset,
                           int remote_num_pages) {
  // launch grid: [num_blocks, 1, 1], [32, local_num_kv_head, 1]
  // pages(remote): [remote_num_pages, 2, remote_num_kv_head, page_size, head_dim]
  // k_data: [ntokens, local_num_kv_head, head_dim]
  // v_data: [ntokens, local_num_kv_head, head_dim]
  int remote_pe;
  int remote_kv_head_index;
  int h = threadIdx.y;  // local kv head index

  for (int global_pos = blockIdx.x; global_pos < ntokens; global_pos += gridDim.x) {
    int position = remote_position_map[global_pos];
    if (position == -1) {
      continue;
    }
    if (local_num_kv_head <= remote_num_kv_head) {
      // gather
      assert(remote_num_kv_head % local_num_kv_head == 0);
      int gather_factor = remote_num_kv_head / local_num_kv_head;
      remote_pe = remote_tp_group_pe_offset[global_pos] + local_tp_rank / gather_factor;
      remote_kv_head_index = (local_tp_rank % gather_factor) * local_num_kv_head + h;
    } else {
      // scatter
      assert(local_num_kv_head % remote_num_kv_head == 0);
      int scatter_factor = local_num_kv_head / remote_num_kv_head;
      remote_pe = remote_tp_group_pe_offset[global_pos] + local_tp_rank * scatter_factor +
                  h / remote_num_kv_head;
      remote_kv_head_index = h % remote_num_kv_head;
    }
    int page_id = position / page_size;
    int offset_in_page = position % page_size;
    int pages_shape[5] = {remote_num_pages, 2, remote_num_kv_head, page_size, head_dim};
    int k_page_index[5] = {page_id, 0, remote_kv_head_index, offset_in_page, 0};
    int v_page_index[5] = {page_id, 1, remote_kv_head_index, offset_in_page, 0};
    int k_v_shape[3] = {ntokens, local_num_kv_head, head_dim};
    int k_v_index[3] = {global_pos, h, 0};
    nvshmemx_putmem_nbi_warp(pages + calc_flattened_index<5>(pages_shape, k_page_index),
                             k_data + calc_flattened_index<3>(k_v_shape, k_v_index),
                             head_dim * sizeof(T), remote_pe);
    nvshmemx_putmem_nbi_warp(pages + calc_flattened_index<5>(pages_shape, v_page_index),
                             v_data + calc_flattened_index<3>(k_v_shape, k_v_index),
                             head_dim * sizeof(T), remote_pe);
  }
  if (threadIdx.x == 0) {
    nvshmem_quiet();
  }
}
template <typename T, int local_num_kv_head, int remote_num_kv_head, int head_dim, int page_size>
__global__ void KVTransferPageToPage(T* remote_pages, T* local_pages, int32_t* remote_position_map,
                                     int32_t* local_position_map, int ntokens, int local_tp_rank,
                                     int32_t* remote_tp_group_pe_offset) {
  // launch grid: [num_blocks, 1, 1], [32, local_num_kv_head, 1]
  int remote_pe;
  int remote_kv_head_index;
  int h = threadIdx.y;  // local kv head index
  int is_k = threadIdx.z;

  for (int global_pos = blockIdx.x; global_pos < ntokens; global_pos += gridDim.x) {
    int remote_position = remote_position_map[global_pos];
    int local_position = local_position_map[global_pos];
    if (remote_position == -1 || local_position == -1) {
      continue;
    }
    if (local_num_kv_head <= remote_num_kv_head) {
      // gather
      assert(remote_num_kv_head % local_num_kv_head == 0);
      int gather_factor = remote_num_kv_head / local_num_kv_head;
      remote_pe = remote_tp_group_pe_offset[global_pos] + local_tp_rank / gather_factor;
      remote_kv_head_index = (local_tp_rank % gather_factor) * local_num_kv_head + h;
    } else {
      // scatter
      assert(local_num_kv_head % remote_num_kv_head == 0);
      int scatter_factor = local_num_kv_head / remote_num_kv_head;
      remote_pe = remote_tp_group_pe_offset[global_pos] + local_tp_rank * scatter_factor +
                  h / remote_num_kv_head;
      remote_kv_head_index = h % remote_num_kv_head;
    }

    int remote_page_id = remote_position / page_size;
    int remote_offset_in_page = remote_position % page_size;
    int local_page_id = local_position / page_size;
    int local_offset_in_page = local_position % page_size;
    int remote_pages_shape[5] = {1, 2, remote_num_kv_head, page_size, head_dim};
    int local_pages_shape[5] = {1, 2, local_num_kv_head, page_size, head_dim};
    int remote_page_index[5] = {remote_page_id, is_k, remote_kv_head_index, remote_offset_in_page,
                                0};
    int local_page_index[5] = {local_page_id, is_k, h, local_offset_in_page, 0};
    nvshmemx_putmem_nbi_warp(
        remote_pages + calc_flattened_index<5>(remote_pages_shape, remote_page_index),
        local_pages + calc_flattened_index<5>(local_pages_shape, local_page_index),
        head_dim * sizeof(T), remote_pe);
  }
  if (threadIdx.x == 0) {
    nvshmem_quiet();
  }
}

#define DISPATCH_TVM_CUDA_DTYPE(dl_dtype, cuda_dtype, ...)   \
  if (dl_dtype.code == kDLFloat && dl_dtype.bits == 16) {    \
    using cuda_dtype = half;                                 \
    __VA_ARGS__                                              \
  } else {                                                   \
    LOG(FATAL) << "Unsupported data type " << dl_dtype.code; \
  }

#define DISPATCH_HEAD_DIM(head_dim, const_head_dim, ...) \
  if (head_dim == 128) {                                 \
    constexpr int const_head_dim = 128;                  \
    __VA_ARGS__                                          \
  } else {                                               \
    LOG(FATAL) << "Unsupported head dim " << head_dim;   \
  }

#define DISPATCH_PAGE_SIZE(page_size, const_page_size, ...) \
  if (page_size == 16) {                                    \
    constexpr int const_page_size = 16;                     \
    __VA_ARGS__                                             \
  } else if (page_size == 4) {                              \
    constexpr int const_page_size = 4;                      \
    __VA_ARGS__                                             \
  } else {                                                  \
    LOG(FATAL) << "Unsupported page size " << page_size;    \
  }

#define DISPATCH_NUM_KV_HEAD(num_kv_head, const_num_kv_head, ...) \
  if (num_kv_head == 1) {                                         \
    constexpr int const_num_kv_head = 1;                          \
    __VA_ARGS__                                                   \
  } else if (num_kv_head == 2) {                                  \
    constexpr int const_num_kv_head = 2;                          \
    __VA_ARGS__                                                   \
  } else if (num_kv_head == 4) {                                  \
    constexpr int const_num_kv_head = 4;                          \
    __VA_ARGS__                                                   \
  } else if (num_kv_head == 8) {                                  \
    constexpr int const_num_kv_head = 8;                          \
    __VA_ARGS__                                                   \
  } else {                                                        \
    LOG(FATAL) << "Unsupported num_kv_head " << num_kv_head;      \
  }

int _KVTransfer(DLTensor* remote_pages, DLTensor* k, DLTensor* v, DLTensor* remote_position_map,
                DLTensor* remote_tp_group_pe_offset, TVMStreamHandle transfer_stream) {
  CHECK_EQ(remote_pages->device.device_type, kDLCUDA)
      << "The device of remote_pages matrix must be CUDA.";
  CHECK_EQ(k->device.device_type, kDLCUDA) << "The device of k matrix must be CUDA.";
  CHECK_EQ(v->device.device_type, kDLCUDA) << "The device of v matrix must be CUDA.";
  CHECK_EQ(remote_position_map->device.device_type, kDLCUDA)
      << "The device of remote_position_map matrix must be CUDA.";
  size_t dev_id = remote_pages->device.device_id;
  CHECK_EQ(k->device.device_id, dev_id)
      << "The device id of remote_pages and k matrix doesn't match.";
  CHECK_EQ(v->device.device_id, dev_id)
      << "The device id of remote_pages and v matrix doesn't match.";
  CHECK_EQ(remote_position_map->device.device_id, dev_id)
      << "The device id of remote_pages and remote_position_map matrix doesn't match.";
  CHECK_EQ(remote_tp_group_pe_offset->device.device_id, dev_id)
      << "The device id of remote_pages and remote_tp_group_pe_offset matrix doesn't match.";

  CHECK_EQ(remote_pages->ndim, 5);
  int remote_num_pages = remote_pages->shape[0];
  int remote_num_kv_head = remote_pages->shape[2];
  int page_size = remote_pages->shape[3];
  int head_dim = remote_pages->shape[4];

  CHECK_GE(k->ndim, 3);
  int kv_len = k->shape[k->ndim - 3];
  int local_num_kv_heads = k->shape[k->ndim - 2];
  CHECK_EQ(head_dim, k->shape[k->ndim - 1]);

  CHECK_GE(v->ndim, 3);
  CHECK_EQ(kv_len, v->shape[v->ndim - 3]);
  CHECK_EQ(local_num_kv_heads, v->shape[v->ndim - 2]);
  CHECK_EQ(head_dim, v->shape[v->ndim - 1]);

  CHECK(remote_pages->dtype.lanes == 1 && k->dtype.lanes == 1 && v->dtype.lanes == 1);
  CHECK(remote_pages->dtype.bits == k->dtype.bits && remote_pages->dtype.code == k->dtype.code);
  CHECK(remote_pages->dtype.bits == v->dtype.bits && remote_pages->dtype.code == v->dtype.code);
  int local_tp_rank;
  tvm::runtime::DiscoWorker* worker = tvm::runtime::ThreadLocalDiscoWorker::Get()->worker;
  if (worker == nullptr) {
    local_tp_rank = 0;
  } else {
    local_tp_rank = worker->worker_id;
  }

  dim3 blocks(8, 1, 1);
  dim3 threads(32, local_num_kv_heads, 1);
  DISPATCH_TVM_CUDA_DTYPE(
      remote_pages->dtype, dtype_in,
      {DISPATCH_HEAD_DIM(
          head_dim, HEAD_DIM,
          {DISPATCH_PAGE_SIZE(
              page_size, PAGE_SIZE,
              {DISPATCH_NUM_KV_HEAD(
                  remote_num_kv_head, REMOTE_NUM_KV_HEAD,
                  {DISPATCH_NUM_KV_HEAD(local_num_kv_heads, LOCAL_NUM_KV_HEAD, {
                    dtype_in* remote_pages_data = reinterpret_cast<dtype_in*>(
                        reinterpret_cast<char*>(remote_pages->data) + remote_pages->byte_offset);
                    dtype_in* k_data = reinterpret_cast<dtype_in*>(
                        reinterpret_cast<char*>(k->data) + k->byte_offset);
                    dtype_in* v_data = reinterpret_cast<dtype_in*>(
                        reinterpret_cast<char*>(v->data) + v->byte_offset);
                    int32_t* remote_position_map_data = reinterpret_cast<int32_t*>(
                        reinterpret_cast<char*>(remote_position_map->data) +
                        remote_position_map->byte_offset);
                    int32_t* remote_tp_group_pe_offset_data = reinterpret_cast<int32_t*>(
                        reinterpret_cast<char*>(remote_tp_group_pe_offset->data) +
                        remote_tp_group_pe_offset->byte_offset);
                    KVTransfer<dtype_in, LOCAL_NUM_KV_HEAD, REMOTE_NUM_KV_HEAD, HEAD_DIM, PAGE_SIZE>
                        <<<blocks, threads, 0, static_cast<cudaStream_t>(transfer_stream)>>>(
                            remote_pages_data, k_data, v_data, remote_position_map_data, kv_len,
                            local_tp_rank, remote_tp_group_pe_offset_data, remote_num_pages);
                  })})})})})

  return 0;
}

int _KVTransferPageToPage(DLTensor* remote_pages, DLTensor* local_pages,
                          DLTensor* remote_position_map, DLTensor* local_position_map,
                          DLTensor* remote_tp_group_pe_offset, TVMStreamHandle transfer_stream) {
  CHECK_EQ(remote_pages->device.device_type, kDLCUDA)
      << "The device of remote_pages matrix must be CUDA.";
  CHECK_EQ(local_pages->device.device_type, kDLCUDA) << "The device of k matrix must be CUDA.";
  CHECK_EQ(remote_position_map->device.device_type, kDLCUDA)
      << "The device of remote_position_map matrix must be CUDA.";
  size_t dev_id = remote_pages->device.device_id;
  CHECK_EQ(local_pages->device.device_id, dev_id)
      << "The device id of remote_pages and k matrix doesn't match.";
  CHECK_EQ(remote_position_map->device.device_id, dev_id)
      << "The device id of remote_pages and remote_position_map matrix doesn't match.";
  CHECK_EQ(remote_tp_group_pe_offset->device.device_id, dev_id)
      << "The device id of remote_pages and remote_tp_group_pe_offset matrix doesn't match.";

  CHECK_EQ(remote_pages->ndim, 5);
  int remote_num_kv_head = remote_pages->shape[2];
  int page_size = remote_pages->shape[3];
  int head_dim = remote_pages->shape[4];

  CHECK_GE(local_pages->ndim, 5);
  int local_num_kv_heads = local_pages->shape[2];
  CHECK_EQ(head_dim, local_pages->shape[4]);

  CHECK_EQ(remote_position_map->ndim, 1);
  int ntokens = remote_position_map->shape[0];

  CHECK(remote_pages->dtype.lanes == 1 && local_pages->dtype.lanes == 1);
  CHECK(remote_pages->dtype.bits == local_pages->dtype.bits &&
        remote_pages->dtype.code == local_pages->dtype.code);

  int local_tp_rank;
  tvm::runtime::DiscoWorker* worker = tvm::runtime::ThreadLocalDiscoWorker::Get()->worker;
  if (worker == nullptr) {
    local_tp_rank = 0;
  } else {
    local_tp_rank = worker->worker_id;
  }

  dim3 blocks(8, 1, 1);
  dim3 threads(32, local_num_kv_heads, 2);
  DISPATCH_TVM_CUDA_DTYPE(
      remote_pages->dtype, dtype_in,
      {DISPATCH_HEAD_DIM(
          head_dim, HEAD_DIM,
          {DISPATCH_PAGE_SIZE(
              page_size, PAGE_SIZE,
              {DISPATCH_NUM_KV_HEAD(
                  remote_num_kv_head, REMOTE_NUM_KV_HEAD,
                  {DISPATCH_NUM_KV_HEAD(local_num_kv_heads, LOCAL_NUM_KV_HEAD, {
                    dtype_in* remote_pages_data = reinterpret_cast<dtype_in*>(
                        reinterpret_cast<char*>(remote_pages->data) + remote_pages->byte_offset);
                    dtype_in* local_pages_data = reinterpret_cast<dtype_in*>(
                        reinterpret_cast<char*>(local_pages->data) + local_pages->byte_offset);
                    int32_t* remote_position_map_data = reinterpret_cast<int32_t*>(
                        reinterpret_cast<char*>(remote_position_map->data) +
                        remote_position_map->byte_offset);
                    int32_t* local_position_map_data = reinterpret_cast<int32_t*>(
                        reinterpret_cast<char*>(local_position_map->data) +
                        local_position_map->byte_offset);
                    int32_t* remote_tp_group_pe_offset_data = reinterpret_cast<int32_t*>(
                        reinterpret_cast<char*>(remote_tp_group_pe_offset->data) +
                        remote_tp_group_pe_offset->byte_offset);
                    KVTransferPageToPage<dtype_in, LOCAL_NUM_KV_HEAD, REMOTE_NUM_KV_HEAD, HEAD_DIM,
                                         PAGE_SIZE>
                        <<<blocks, threads, 0, static_cast<cudaStream_t>(transfer_stream)>>>(
                            remote_pages_data, local_pages_data, remote_position_map_data,
                            local_position_map_data, ntokens, local_tp_rank,
                            remote_tp_group_pe_offset_data);
                  })})})})})

  return 0;
}

TVM_REGISTER_GLOBAL("nvshmem.KVTransfer").set_body_typed(_KVTransfer);
TVM_REGISTER_GLOBAL("nvshmem.KVTransferPageToPage").set_body_typed(_KVTransferPageToPage);
