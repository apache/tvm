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

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/container/shape_tuple.h>

#include "../../../3rdparty/libflash_attn/include/flash.h"

namespace tvm {
namespace runtime {

/*
  query: (batch_size, seqlen_q, num_heads, head_size), fp16
  key_cache: (num_blocks, page_block_size, num_heads_k, head_size), fp16
  value_cache: num_blocks, page_block_size, num_heads_k, head_size), fp16
  block_tables: (batch_size, max_num_blocks_per_seq), int32
  context_lens: (batch_size,), int32
  softmax_lse_accum: (max_num_splits, batch_size, num_heads, seqlen_q), fp32
  output_accum: (max_num_splits, batch_size, num_heads, seqlen_q, head_size), fp32
  out: (batch_size, seqlen_q, num_heads, head_size), fp16
*/
TVM_REGISTER_GLOBAL("tvm.contrib.flash_attn.flash_decoding_with_paged_kvcache")
  .set_body_typed([](const DLTensor* query, const DLTensor* key_cache,
                     const DLTensor* value_cache, const DLTensor* block_tables,
                     const DLTensor* context_lens, DLTensor* softmax_lse_accum,
                     DLTensor* output_accum, DLTensor* out) {
      int batch_size = query->shape[0];
      int seqlen_q = query->shape[1];
      int num_heads = query->shape[2];
      int head_dim = query->shape[3];
      int num_heads_k = key_cache->shape[2];
      int num_blocks = key_cache->shape[0];
      int block_size = key_cache->shape[1];
      int max_num_blocks_per_seq = block_tables->shape[1];
      float softmax_scale = 1.0 / sqrt(static_cast<float>(head_dim));

      ICHECK(block_size % 256 == 0) << "Block size needs to be a multiple of 256.";

      auto block_table_ptr = static_cast<int*>(block_tables->data);
      auto seqlens_k_ptr = static_cast<int*>(context_lens->data);

      using half = ::flash_attn::half;

      ICHECK(TypeMatch(block_tables->dtype, kDLInt, 32));
      ICHECK(TypeMatch(context_lens->dtype, kDLInt, 32));
      ICHECK(TypeMatch(softmax_lse_accum->dtype, kDLFloat, 32));
      ICHECK(TypeMatch(output_accum->dtype, kDLFloat, 32));

      auto q_ptr = static_cast<half*>(query->data);
      auto kcache_ptr = static_cast<half*>(key_cache->data);
      auto vcache_ptr = static_cast<half*>(value_cache->data);
      auto softmax_lse_accum_ptr = static_cast<float*>(softmax_lse_accum->data);
      auto output_accum_ptr = static_cast<float*>(output_accum->data);
      auto output_ptr = static_cast<half*>(out->data);

      int q_head_stride = head_dim;
      int k_head_stride = head_dim;
      int v_head_stride = head_dim;
      int o_head_stride = head_dim;
      int q_row_stride = q_head_stride * num_heads;
      int k_row_stride = k_head_stride * num_heads_k;
      int v_row_stride = v_head_stride * num_heads_k;
      int o_row_stride = o_head_stride * num_heads;
      int q_batch_stride = q_row_stride * seqlen_q;
      int k_batch_stride = k_row_stride * block_size;
      int v_batch_stride = v_row_stride * block_size;
      int o_batch_stride = o_row_stride * seqlen_q;
      int block_table_batch_stride = max_num_blocks_per_seq;

      ::flash_attn::flash_attention_splitkv_paged_forward(
          q_ptr, kcache_ptr, vcache_ptr, block_table_ptr, seqlens_k_ptr,
          softmax_lse_accum_ptr, output_accum_ptr,
          output_ptr, batch_size, seqlen_q, num_heads, num_heads_k, head_dim,
          q_batch_stride,
          k_batch_stride,
          v_batch_stride,
          o_batch_stride,
          q_head_stride,
          k_head_stride,
          v_head_stride,
          o_head_stride,
          q_row_stride,
          k_row_stride,
          v_row_stride,
          o_row_stride,
          num_blocks, block_size, max_num_blocks_per_seq,
          block_table_batch_stride,
          softmax_scale,
          true /* is_causal*/);
    });

}  // namespace runtime
}  // namespace tvm
