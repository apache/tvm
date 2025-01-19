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
namespace relax_vm {

// Register Object Type
TVM_REGISTER_OBJECT_TYPE(KVStateObj);
TVM_REGISTER_OBJECT_TYPE(AttentionKVCacheObj);
TVM_REGISTER_OBJECT_TYPE(RNNStateObj);

// KV State base methods
TVM_REGISTER_GLOBAL("vm.builtin.kv_state_clear").set_body_method<KVState>(&KVStateObj::Clear);
TVM_REGISTER_GLOBAL("vm.builtin.kv_state_add_sequence")
    .set_body_method<KVState>(&KVStateObj::AddSequence);
TVM_REGISTER_GLOBAL("vm.builtin.kv_state_remove_sequence")
    .set_body_method<KVState>(&KVStateObj::RemoveSequence);
TVM_REGISTER_GLOBAL("vm.builtin.kv_state_fork_sequence")
    .set_body_method<KVState>(&KVStateObj::ForkSequence);
TVM_REGISTER_GLOBAL("vm.builtin.kv_state_popn").set_body_method<KVState>(&KVStateObj::PopN);
TVM_REGISTER_GLOBAL("vm.builtin.kv_state_begin_forward")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.size() == 3 || args.size() == 4)
          << "KVState BeginForward only accepts 3 or 4 arguments";
      KVState kv_state = args[0];
      IntTuple seq_ids = args[1];
      IntTuple append_lengths = args[2];
      Optional<IntTuple> token_tree_parent_ptr{nullptr};
      if (args.size() == 4) {
        token_tree_parent_ptr = args[3].operator Optional<IntTuple>();
      }
      kv_state->BeginForward(seq_ids, append_lengths, token_tree_parent_ptr);
    });
TVM_REGISTER_GLOBAL("vm.builtin.kv_state_end_forward")
    .set_body_method<KVState>(&KVStateObj::EndForward);

// Attention KV Cache methods
TVM_REGISTER_GLOBAL("vm.builtin.kv_cache_disagg_prepare_recv")
    .set_body_method<AttentionKVCache>(&AttentionKVCacheObj::DisaggPrepareRecv);
TVM_REGISTER_GLOBAL("vm.builtin.kv_cache_disagg_mark_send")
    .set_body_method<AttentionKVCache>(&AttentionKVCacheObj::DisaggMarkSend);
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_enable_sliding_window_for_seq")
    .set_body_method<AttentionKVCache>(&AttentionKVCacheObj::EnableSlidingWindowForSeq);
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_commit_accepted_token_tree_nodes")
    .set_body_method<AttentionKVCache>(&AttentionKVCacheObj::CommitAcceptedTokenTreeNodes);
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_empty")
    .set_body_method<AttentionKVCache>(&AttentionKVCacheObj::Empty);
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_get_num_available_pages")
    .set_body_method<AttentionKVCache>(&AttentionKVCacheObj::GetNumAvailablePages);
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_get_total_sequence_length")
    .set_body_method<AttentionKVCache>(&AttentionKVCacheObj::GetTotalSequenceLength);
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_get_query_positions")
    .set_body_method<AttentionKVCache>(&AttentionKVCacheObj::GetQueryPositions);
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_debug_get_kv")
    .set_body_method<AttentionKVCache>(&AttentionKVCacheObj::DebugGetKV);
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_debug_get_kv_mla")
    .set_body_method<AttentionKVCache>(&AttentionKVCacheObj::DebugGetKVMLA);
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_attention_with_fused_qkv")
    .set_body_typed([](AttentionKVCache kv_cache, int64_t layer_id,
                       double attn_score_scaling_factor, NDArray qkv_data, NDArray o_data) {
      kv_cache->AttentionWithFusedQKV(layer_id, std::move(qkv_data), NullOpt, std::move(o_data),
                                      attn_score_scaling_factor);
    });
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_mla_absorbed")
    .set_body_typed([](AttentionKVCache kv_cache, int64_t layer_id,
                       double attn_score_scaling_factor, NDArray q_data, NDArray compressed_kv_data,
                       NDArray k_pe_data, NDArray o_data) {
      kv_cache->MLAAbsorbed(layer_id, std::move(q_data), std::move(compressed_kv_data),
                            std::move(k_pe_data), std::move(o_data), attn_score_scaling_factor);
    });

// RNN State methods
TVM_REGISTER_GLOBAL("vm.builtin.rnn_state_get").set_body_method<RNNState>(&RNNStateObj::Get);
TVM_REGISTER_GLOBAL("vm.builtin.rnn_state_set")
    .set_body_typed([](RNNState state, int64_t layer_id, int64_t state_id, NDArray data) {
      state->Set(layer_id, state_id, data);
      return state;
    });
TVM_REGISTER_GLOBAL("vm.builtin.rnn_state_debug_get")
    .set_body_method<RNNState>(&RNNStateObj::DebugGet);

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
