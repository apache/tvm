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

#include <tvm/ffi/reflection/registry.h>

#include <utility>

namespace tvm {
namespace runtime {
namespace vm {

// Register Object Type

// KV State base methods
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("vm.builtin.kv_state_clear", &KVStateObj::Clear)
      .def_method("vm.builtin.kv_state_add_sequence", &KVStateObj::AddSequence)
      .def_method("vm.builtin.kv_state_remove_sequence", &KVStateObj::RemoveSequence)
      .def_method("vm.builtin.kv_state_fork_sequence", &KVStateObj::ForkSequence)
      .def_method("vm.builtin.kv_state_popn", &KVStateObj::PopN)
      .def_packed("vm.builtin.kv_state_begin_forward",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    CHECK(args.size() == 4 || args.size() == 5)
                        << "KVState BeginForward only accepts 4 or 5 arguments";
                    KVState kv_state = args[0].cast<KVState>();
                    ffi::Shape seq_ids = args[1].cast<ffi::Shape>();
                    ffi::Shape append_lengths = args[2].cast<ffi::Shape>();
                    int64_t seqlen_padding_factor = args[3].cast<int64_t>();
                    Optional<ffi::Shape> token_tree_parent_ptr;
                    if (args.size() == 5) {
                      token_tree_parent_ptr = args[4].cast<Optional<ffi::Shape>>();
                    }
                    kv_state->BeginForward(seq_ids, append_lengths, seqlen_padding_factor, token_tree_parent_ptr);
                  })
      .def_method("vm.builtin.kv_state_end_forward", &KVStateObj::EndForward);
});

// Attention KV Cache methods
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("vm.builtin.kv_cache_disagg_prepare_recv",
                  &AttentionKVCacheObj::DisaggPrepareRecv)
      .def_method("vm.builtin.kv_cache_disagg_mark_send", &AttentionKVCacheObj::DisaggMarkSend)
      .def_method("vm.builtin.attention_kv_cache_enable_sliding_window_for_seq",
                  &AttentionKVCacheObj::EnableSlidingWindowForSeq)
      .def_method("vm.builtin.attention_kv_cache_commit_accepted_token_tree_nodes",
                  &AttentionKVCacheObj::CommitAcceptedTokenTreeNodes)
      .def_method("vm.builtin.attention_kv_cache_empty", &AttentionKVCacheObj::Empty)
      .def_method("vm.builtin.attention_kv_cache_get_num_available_pages",
                  &AttentionKVCacheObj::GetNumAvailablePages)
      .def_method("vm.builtin.attention_kv_cache_get_total_sequence_length",
                  &AttentionKVCacheObj::GetTotalSequenceLength)
      .def_method("vm.builtin.attention_kv_cache_get_query_positions",
                  &AttentionKVCacheObj::GetQueryPositions)
      .def_method("vm.builtin.attention_kv_cache_debug_get_kv", &AttentionKVCacheObj::DebugGetKV)
      .def_method("vm.builtin.attention_kv_cache_debug_get_kv_mla",
                  &AttentionKVCacheObj::DebugGetKVMLA)
      .def("vm.builtin.attention_kv_cache_attention_with_fused_qkv",
           [](AttentionKVCache kv_cache, int64_t layer_id, double sm_scale, NDArray qkv_data,
              NDArray o_data) {
             kv_cache->AttentionWithFusedQKV(layer_id, std::move(qkv_data), std::nullopt,
                                             std::move(o_data), sm_scale);
           })
      .def("vm.builtin.attention_kv_cache_self_attention",
           [](AttentionKVCache kv_cache, int64_t layer_id, double sm_scale, NDArray q_data,
              NDArray k_data, NDArray v_data, NDArray o_data, NDArray lse_data) {
             kv_cache->SelfAttention(layer_id, std::move(q_data), std::move(k_data),
                                     std::move(v_data), std::move(o_data), std::move(lse_data),
                                     sm_scale);
           })
      .def("vm.builtin.attention_kv_cache_cross_attention",
           [](AttentionKVCache kv_cache, int64_t layer_id, double sm_scale, NDArray q_data,
              NDArray o_data, NDArray lse_data) {
             kv_cache->CrossAttention(layer_id, std::move(q_data), std::move(o_data),
                                      std::move(lse_data), sm_scale);
           })
      .def("vm.builtin.attention_kv_cache_append_mla_kv",
           [](AttentionKVCache kv_cache, int64_t layer_id, NDArray kv_data) {
             kv_cache->AppendMLAKV(layer_id, std::move(kv_data));
             return kv_cache;
           })
      .def("vm.builtin.attention_kv_cache_merge_attn_output_inplace",
           [](AttentionKVCache kv_cache, NDArray o_self_attn, NDArray lse_self_attn,
              NDArray o_cross_attn, NDArray lse_cross_attn) {
             return kv_cache->MergeAttnOutputInplace(
                 std::move(o_self_attn), std::move(lse_self_attn), std::move(o_cross_attn),
                 std::move(lse_cross_attn));
           });
});

// RNN State methods
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("vm.builtin.rnn_state_get", &RNNStateObj::Get)
      .def("vm.builtin.rnn_state_set",
           [](RNNState state, int64_t layer_id, int64_t state_id, NDArray data) {
             state->Set(layer_id, state_id, data);
             return state;
           })
      .def_method("vm.builtin.rnn_state_debug_get", &RNNStateObj::DebugGet);
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
