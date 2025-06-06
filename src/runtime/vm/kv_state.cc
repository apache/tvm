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

#include "kv_state.h"

#include <utility>

namespace tvm {
namespace runtime {
namespace vm {

// Register Object Type
TVM_REGISTER_OBJECT_TYPE(KVStateObj);
TVM_REGISTER_OBJECT_TYPE(AttentionKVCacheObj);
TVM_REGISTER_OBJECT_TYPE(RNNStateObj);

// KV State base methods
TVM_FFI_REGISTER_GLOBAL("vm.builtin.kv_state_clear").set_body_method(&KVStateObj::Clear);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.kv_state_add_sequence")
    .set_body_method(&KVStateObj::AddSequence);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.kv_state_remove_sequence")
    .set_body_method(&KVStateObj::RemoveSequence);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.kv_state_fork_sequence")
    .set_body_method(&KVStateObj::ForkSequence);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.kv_state_popn").set_body_method(&KVStateObj::PopN);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.kv_state_begin_forward")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      CHECK(args.size() == 3 || args.size() == 4)
          << "KVState BeginForward only accepts 3 or 4 arguments";
      KVState kv_state = args[0].cast<KVState>();
      ffi::Shape seq_ids = args[1].cast<ffi::Shape>();
      ffi::Shape append_lengths = args[2].cast<ffi::Shape>();
      Optional<ffi::Shape> token_tree_parent_ptr;
      if (args.size() == 4) {
        token_tree_parent_ptr = args[3].cast<Optional<ffi::Shape>>();
      }
      kv_state->BeginForward(seq_ids, append_lengths, token_tree_parent_ptr);
    });
TVM_FFI_REGISTER_GLOBAL("vm.builtin.kv_state_end_forward").set_body_method(&KVStateObj::EndForward);

// Attention KV Cache methods
TVM_FFI_REGISTER_GLOBAL("vm.builtin.kv_cache_disagg_prepare_recv")
    .set_body_method(&AttentionKVCacheObj::DisaggPrepareRecv);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.kv_cache_disagg_mark_send")
    .set_body_method(&AttentionKVCacheObj::DisaggMarkSend);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_enable_sliding_window_for_seq")
    .set_body_method(&AttentionKVCacheObj::EnableSlidingWindowForSeq);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_commit_accepted_token_tree_nodes")
    .set_body_method(&AttentionKVCacheObj::CommitAcceptedTokenTreeNodes);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_empty")
    .set_body_method(&AttentionKVCacheObj::Empty);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_get_num_available_pages")
    .set_body_method(&AttentionKVCacheObj::GetNumAvailablePages);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_get_total_sequence_length")
    .set_body_method(&AttentionKVCacheObj::GetTotalSequenceLength);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_get_query_positions")
    .set_body_method(&AttentionKVCacheObj::GetQueryPositions);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_debug_get_kv")
    .set_body_method(&AttentionKVCacheObj::DebugGetKV);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_debug_get_kv_mla")
    .set_body_method(&AttentionKVCacheObj::DebugGetKVMLA);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_attention_with_fused_qkv")
    .set_body_typed([](AttentionKVCache kv_cache, int64_t layer_id, double sm_scale,
                       NDArray qkv_data, NDArray o_data) {
      kv_cache->AttentionWithFusedQKV(layer_id, std::move(qkv_data), std::nullopt,
                                      std::move(o_data), sm_scale);
    });
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_self_attention")
    .set_body_typed([](AttentionKVCache kv_cache, int64_t layer_id, double sm_scale, NDArray q_data,
                       NDArray k_data, NDArray v_data, NDArray o_data, NDArray lse_data) {
      kv_cache->SelfAttention(layer_id, std::move(q_data), std::move(k_data), std::move(v_data),
                              std::move(o_data), std::move(lse_data), sm_scale);
    });
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_cross_attention")
    .set_body_typed([](AttentionKVCache kv_cache, int64_t layer_id, double sm_scale, NDArray q_data,
                       NDArray o_data, NDArray lse_data) {
      kv_cache->CrossAttention(layer_id, std::move(q_data), std::move(o_data), std::move(lse_data),
                               sm_scale);
    });
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_append_mla_kv")
    .set_body_typed([](AttentionKVCache kv_cache, int64_t layer_id, NDArray kv_data) {
      kv_cache->AppendMLAKV(layer_id, std::move(kv_data));
      return kv_cache;
    });
TVM_FFI_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_merge_attn_output_inplace")
    .set_body_typed([](AttentionKVCache kv_cache, NDArray o_self_attn, NDArray lse_self_attn,
                       NDArray o_cross_attn, NDArray lse_cross_attn) {
      return kv_cache->MergeAttnOutputInplace(std::move(o_self_attn), std::move(lse_self_attn),
                                              std::move(o_cross_attn), std::move(lse_cross_attn));
    });

// RNN State methods
TVM_FFI_REGISTER_GLOBAL("vm.builtin.rnn_state_get").set_body_method(&RNNStateObj::Get);
TVM_FFI_REGISTER_GLOBAL("vm.builtin.rnn_state_set")
    .set_body_typed([](RNNState state, int64_t layer_id, int64_t state_id, NDArray data) {
      state->Set(layer_id, state_id, data);
      return state;
    });
TVM_FFI_REGISTER_GLOBAL("vm.builtin.rnn_state_debug_get").set_body_method(&RNNStateObj::DebugGet);

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
