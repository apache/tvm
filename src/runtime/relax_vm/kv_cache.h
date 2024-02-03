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
#ifndef TVM_RUNTIME_RELAX_VM_KV_CACHE_H_
#define TVM_RUNTIME_RELAX_VM_KV_CACHE_H_
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {
namespace relax_vm {

/*!
 * \brief The base class of attention KV cache for efficient
 * k/v data management and attention computation.
 */
class AttentionKVCache : public Object {
 public:
  /*! \brief Reset the KV cache. */
  virtual void Clear() = 0;

  /************** Sequence Management **************/

  /*!
   * \brief Add a new sequence with empty K/V data in the cache.
   * Check if the validity of the input sequence id.
   * \param seq_id The id of the new sequence to be added.
   * \throws Error if the given sequence id is not valid.
   */
  virtual void AddSequence(int64_t seq_id) = 0;

  /*!
   * \brief Remove a sequence and its K/V data from the KV cache.
   * \param seq_id The sequence to remove from cache.
   * \throws Error if the given sequence id is not valid.
   */
  virtual void RemoveSequence(int64_t seq_id) = 0;

  /*!
   * \brief Fork the K/V data of parent sequence to the child sequence.
   * After the fork, the child sequence has K/V data of the parent
   * sequence.
   * \param parent_seq_id The parent (source) of the fork.
   * \param child_seq_id The child (destination) of the fork.
   * The child sequence id should not exist in cache prior to fork.
   * \throws Error if the given sequence ids are not valid.
   */
  virtual void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id) = 0;

  /*!
   * \brief Pop out the trailing `n` tokens from the KV cache for the
   * specified sequence.
   * \param seq_id The sequence whose trailing tokens are to be popped.
   * \param n The number of tokens to pop from the KV cache.
   * \throws Error if the given sequence id is not valid.
   */
  virtual void PopN(int64_t seq_id, int32_t n) = 0;

  /************** Raw Info Query **************/

  /*!
   * \brief Get the number of available pages in the KV cache.
   * When the underlying KV cache implementation is not
   * paged KV cache, the function falls back to return the
   * number of remaining size (in terms of number of tokens).
   */
  virtual int32_t GetNumAvailablePages() const = 0;

  /************** Attention **************/

  /*!
   * \brief Mark the start of the forward function with the ids of
   * the sequences and the sequence length to forward for each
   * sequence.
   * For example, if we want to prefill 3 sequences "5", "1", "8"
   * with prefill length "10", "15", "20", then we pass `[5, 1, 8]`
   * as the seq_ids and `[10, 15, 20]` as the append_lengths.
   * This method is invoked right before entering the model forward
   * function, and contains operations to prepare the the incoming
   * forward. For instance, this method may send auxiliary KV cache
   * data structures to GPUs so that they can be operated
   * in the model forward function.
   * \param seq_ids The ids of the sequence to run in the incoming model forward.
   * \param append_lengths The sequence lengths to run forward for for each sequence.
   */
  virtual void BeginForward(const IntTuple& seq_ids, const IntTuple& append_lengths) = 0;

  /*!
   * \brief Mark the start of the forward function.
   * This method is invoked right after entering the model forward
   * function, and contains post-processing of the forward.
   */
  virtual void EndForward() = 0;

  /*!
   * \brief Compute attention with the given Q/K/V data at the specified
   * layer with regard to the previously reserved append lengths.
   * Q/K/V data are in layout `(total_length, num_heads, head_dim)`,
   * where `total_length` is the sum of reserved append lengths.
   * The returned attention result has the same layout as well.
   * For example, say the KV cache contains 5 sequences. Before
   * the current model forward, BeginForward is invoked for seq_ids
   * `[3, 2]` and append_lengths [10, 20]. Then the leading dim of Q/K/V
   * is 30, where [0, 10) corresponds to seq 3, and [10, 30)
   * corresponds to seq 2.
   * This method typically performs the following operations:
   * - apply positional embeddings to Q/K data,
   * - append K/V data to cache,
   * - compute attention with the given Q and all history K/V
   * for the corresponding sequences.
   * The function writes attention output to `o_data`, conforming to
   * the destination-passing style.
   * \param layer_id The model layer where the attention compute happens.
   * \param q_data The input Q data, in layout `(total_length, num_qo_heads, head_dim)`.
   * \param k_data The input K data, in layout `(total_length, num_kv_heads, head_dim)`.
   * \param v_data The input V data, in layout `(total_length, num_kv_heads, head_dim)`.
   * \param mask The input mask data, in layout `(total_sqr_length)`.
   * \param o_data The output O data, in layout `(total_length, num_qo_heads, head_dim)`.
   */
  virtual void Attention(int64_t layer_id, NDArray q_data, NDArray k_data, NDArray v_data,
                         Optional<NDArray> mask, NDArray o_data) = 0;

  /*!
   * \brief Compute attention with Q/K/V data which are concatenated along
   * the head dimension.
   * \param layer_id The model layer where the attention compute happens.
   * \param qkv_data The input Q/K/V data, in layout
   * `(total_length, num_qo_heads + 2 * num_kv_heads, head_dim)`.
   * \param mask The input mask data, in layout `(total_sqr_length)`.
   * \param o_data The output O data, in layout `(total_length, num_qo_heads, head_dim)`.
   * \sa AttentionKVCache::Attention
   */
  virtual void AttentionWithFusedQKV(int64_t layer_id, NDArray qkv_data, Optional<NDArray> mask,
                                     NDArray o_data) = 0;

  /************** Debug Helpers **************/

  /*!
   * \brief Fetch the compact K/V data of the given sequence.
   * The results are written into `k_data` and `v_data` passed in,
   * conforming to the destination-passing style.
   * `start_pos` (inclusive) and `end_pos` (exclusive) controls the range
   * of K/V data to fetch.
   * - `start_pos` defaults to 0 when undefined,
   * - `end_pos` defaults to the length of sequence when undefined.
   * This method expects `start_pos < end_pos`.
   * The k/v data arrays are expected to have layout
   * `(num_layers, start_pos - end_pos, num_heads, head_dim)`.
   * \param seq_id The sequence whose K/V data is to be fetched.
   * \param start_pos The start position (inclusive) of the K/V data to fetch.
   * \param end_pos The end position (exclusive) of the K/V data to fetch.
   * \param K_data The output K data of the given sequence in layout elaborated above.
   * \param V_data The output V data of the given sequence in layout elaborated above.
   */
  virtual void DebugGetKV(int64_t seq_id,  //
                          int64_t start_pos, int64_t end_pos, NDArray k_data, NDArray v_data) = 0;

  /*!
   * \brief Set the K/V data of the given sequence from input K/V data.
   * `start_pos` (inclusive) controls starting position of K/V data
   * to set, and defaults to 0 when undefined.
   * The input K/V data is in layout `(num_layers, length, num_heads, head_dim)`,
   * where `length` is the length to set. It implies that the range
   * of set is `[start_pos, start_pos + length)`.
   * It is not allowed that `start_pos + length` exceeds the current
   * length of the given sequence.
   * \param seq_id The sequence whose K/V data is to be set.
   * \param start_pos The start position (inclusive) of the K/V data to set.
   * \param k_data The K data to set in layout elaborated above.
   * \param v_data The V data to set in layout elaborated above.
   */
  virtual void DebugSetKV(int64_t seq_id, int64_t start_pos, NDArray k_data, NDArray v_data) = 0;
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_RELAX_VM_KV_CACHE_H_
