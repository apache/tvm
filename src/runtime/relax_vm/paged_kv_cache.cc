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
/*!
 * \file src/runtime/relax_vm/paged_kv_cache.cc
 * \brief Runtime paged KV cache object for language models.
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <unordered_map>
#include <utility>
#include <vector>

#include "kv_state.h"

namespace tvm {
namespace runtime {
namespace relax_vm {

//-------------------------------------------
// We keep the implementation private as
// they may subject to future changes.
//
// Users can interact with it through the
// runtime API function calls
//-------------------------------------------

/*!
 * \brief The maximum allowed block depth (a.k.a. number of common
 * prefixes) in paged KV cache.
 */
constexpr const int kPagedKVCacheMaxBlockDepth = 5;
/*! \brief The 8MB workspace size for attention auxiliary data. */
constexpr const int kAttnWorkspaceByte = 8 * 1024 * 1024;
/*! \brief The id of the temporary logical page, which is useful for sliding window. */
constexpr const int kPagedKVCacheTempPageId = -1;

/*!
 * \brief The block structure in paged KV cache with common prefix support.
 * Each block contains a list of pages for cached KV data.
 * If a block has `n` pages, the first `n - 1` pages must be
 * full, and only the last page can be partially filled.
 *
 * To support common prefix, each sequence in KV cache is represented
 * as one or more blocks, where the common prefix is a standalone
 * block among.
 *
 * Each block has a parent block when it uses a prefix.
 */
struct Block {
  /*!
   * \brief The ids of the pages in the block.
   * Each page can only be used by a unique block (in other
   * words, different blocks do not share pages).
   */
  std::vector<int32_t> page_ids;
  /*! \brief The total sequence length in the block. */
  int32_t seq_length = 0;
  /*!
   * \brief The start position in sequence of this block.
   * This is the absolute position in the sequence for RoPE computation.
   */
  int32_t start_pos = 0;
  /*!
   * \brief The current attention sink length of the block.
   * It means the the **first** sink size elements will be pinned
   * in the KV cache even when sliding window is enabled.
   */
  int32_t sink_length = 0;
  /*!
   * \brief The start offset of the sliding window in the block.
   * It is always 0 when sliding window attn is not enabled.
   */
  int32_t sliding_window_offset = 0;

  /*! \brief The global index of the block. */
  const int32_t index;
  /*!
   * \brief The global index of the parent block of this block, or -1
   * if the block does not have a parent. */
  int32_t parent_idx = -1;
  /*!
   * \brief The external reference counter of the block.
   * When a block is externally referred by some block,
   * we do not allow appending new KV values to this block.
   */
  int external_ref_cnt = 0;

  explicit Block(int32_t index) : index(index) {}

  /*! \brief Reset the block data. */
  void Reset() {
    page_ids.clear();
    seq_length = 0;
    parent_idx = -1;
    external_ref_cnt = 0;
  }
};

/*!
 * \brief The sequence structure in paged KV cache with common prefix support.
 * Each sequence contains one or more blocks to support common prefix.
 */
struct Sequence {
  /*!
   * \brief The global index of the last block of the sequence.
   * We only store the last block, since all the blocks can be
   * tracked with the `parent` field of Block.
   */
  int32_t last_block_idx;
  /*!
   * \brief The total sequence length of the sequence.
   * It is the sum of lengths of all its blocks.
   */
  int32_t seq_length = 0;
  /*!
   * \brief The sliding window size of the sequence, or -1 if sliding window is not enabled.
   * When a sequence is enabled for sliding window, it can no longer be forked.
   */
  int sliding_window_size = -1;
  /*!
   * \brief The attention sink size of the last block of the sequence.
   * The **first** sink size elements of the last block will be pinned
   * in the KV cache even when sliding window is enabled.
   */
  int last_block_attn_sink_size = 0;

  explicit Sequence(const std::vector<Block>& global_block_pool, int32_t last_block_idx) {
    this->last_block_idx = last_block_idx;
    int32_t block_ptr = last_block_idx;
    // Go through each block in the sequence, sum up the length.
    int depth = 0;
    while (true) {
      const Block& block = global_block_pool[block_ptr];
      this->seq_length += block.seq_length;
      ++depth;
      if (block.parent_idx == -1) {
        break;
      }
      block_ptr = block.parent_idx;
    }
    CHECK_LE(depth, kPagedKVCacheMaxBlockDepth)
        << "Paged KV cache supports one sequence to reuse " << kPagedKVCacheMaxBlockDepth
        << " prefixes (the fork depth) at most. However, the given sequence has fork depth "
        << depth;
  }

  std::vector<int32_t> GetBlockTrace(const std::vector<Block>& global_block_pool) const {
    std::vector<int32_t> trace;
    // Get the trace from the last block of the sequence to the root block.
    int32_t block_ptr = last_block_idx;
    while (block_ptr != -1) {
      trace.push_back(block_ptr);
      block_ptr = global_block_pool[block_ptr].parent_idx;
    }
    // Reverse the trace so that it starts from the root block.
    std::reverse(trace.begin(), trace.end());
    return trace;
  }
};

/*!
 * \brief The rotary embedding mode adopted by the paged KV cache
 * when computing attention.
 * "None" means RoPE is never applied to q and k.
 * "Normal" means RoPE is computed in a standalone kernel.
 * "Inline" means RoPE is computed on-the-fly in attention kernels.
 */
enum class RoPEMode : int {
  kNone = 0,
  kNormal = 1,
  kInline = 2,
};

/*!
 * \brief The paged KV cache for attention.
 * - It supports managing the K/V data of **multiple sequences**.
 * - It manages K/V values by doing paging along the sequence-length
 * dimension with a configured page size.
 * - To add a sequence to the cache, use AddSequence with a provided
 * unique integer sequence id.
 * - The basic example use of the paged KV cache after initialization
 * in each round of model forwarding is the following:
 *   - step 1. use `BeginForward` to specify the list of sequence ids
 *     together with the lengths of append,
 *   - step 2. use `Attention` to pass in the q/k/v values regarding
 *     the sequences and lengths specified in `BeginForward`. The
 *     attention is computed between input queries and the history
 *     key/values plus the input key/values. The input key/values
 *     will be added into the KV cache as well.
 *   - step 3. use `EndForward` to mark the end of forwarding this round.
 *     After calling `EndForward`, it is required to call `BeginForward`
 *     before calling any `Attention`.
 */
class PagedAttentionKVCacheObj : public AttentionKVCacheObj {
 private:
  /********************* Configuration *********************/

  /*! \brief The page size (the sequence length each page manages) of the cache. */
  const int64_t page_size_;
  /*! \brief The number of layers in the model. */
  const int64_t num_layers_;
  /*! \brief The number of query/output heads in the model. */
  const int64_t num_qo_heads_;
  /*! \brief The number of key/value heads in the model. */
  const int64_t num_kv_heads_;
  /*! \brief The number of features each head has. */
  const int64_t head_dim_;
  /*! \brief The number of total pages allocated in KV cache. */
  const int64_t num_total_pages_;
  /*! \brief The maximum total sequence length in a prefill. */
  const int64_t prefill_chunk_size_;
  /*! \brief A boolean flag indicating if the KV cache supports sliding window. */
  const bool support_sliding_window_;

  /*! \brief The RoPE application mode of KV cache.*/
  const RoPEMode rope_mode_;
  /*! \brief The RoPE scale. */
  const double rotary_scale_;
  /*! \brief The RoPE theta. */
  const double rotary_theta_;

  /*! \brief We fix int32 to be the index dtype of auxiliary data. */
  const DLDataType dtype_aux_ = DLDataType(DataType::Int(32, 1));

  /********************* Page Structures *********************/

  /*!
   * \brief The KV data managed by the KV cache.
   * The array has `num_layers` NDArrays, each of them
   * has layout (num_pages, 2, num_heads, page_size, head_dim).
   * Along on the "2" dimension, index 0 stands for K and 1 stands for V.
   */
  Array<NDArray> pages_;
  /*! \brief The list of ids of released pages for page reuse. */
  std::vector<int32_t> free_page_ids_;
  /*! \brief The mapping from sequence ids to sequences. */
  std::unordered_map<int64_t, Sequence> seq_map_;

  /********************* Sequence Block Structures *********************/

  /*! \brief The list of all blocks once allocated. */
  std::vector<Block> global_block_pool_;
  /*! \brief The list of free available blocks (in their indices). */
  std::vector<int32_t> free_block_idx_;

  /*********** Current Batch Info & Auxiliary Arrays on Device ***********/
  //-------------------------------------------
  // The following fields are auxiliary arrays on device.
  // All of them are directly derivable from the fields above.
  // We store them for efficient execution of attentions,
  // cache append, etc.
  //-------------------------------------------
  /*!
   * \brief A boolean flag indicating if the auxiliary arrays are dirty.
   * If it is dirty, an explicit "ComputeStreamWaitForCopyStream" should be invoked.
   */
  bool dirty_aux_data_device_ = false;
  /*! \brief The batch size of the current round of forwarding. */
  int64_t cur_batch_size_;
  /*! \brief The append lengths of the sequences in the current round of forwarding. */
  IntTuple cur_append_lengths_;
  /*! \brief The indptr array of append lengths after coalescing. (see GetChunkedBlockIds) */
  std::vector<NDArray> qo_indptr_on_depths_device_;
  /*! \brief The indptr array of page table. */
  std::vector<NDArray> page_indptr_on_depths_device_;
  /*! \brief The indices array of page table. */
  std::vector<NDArray> page_indices_on_depths_device_;
  /*!
   * \brief The length information of the sequences.
   * Each NDArray is in shape `(3, n)`. "n" is the number of sequences.
   * For a sequence "i", location
   * - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
   * - "(1, i)" is the starting offset of the sliding window in the seq,
   * - "(2, i)" is the attn sink length of the sequence.
   * \note When sliding window is not enabled, only the
   * "last_page_len" (a.k.a., the first "n" elements) will be effectively used.
   */
  std::vector<NDArray> length_info_on_depths_device_;
  /*! \brief The k position offset of applying RoPE for each sequence. */
  std::vector<NDArray> k_rope_pos_offset_device_;
  /*!
   * \brief The append length indptr array on device.
   * \note Since the Q/K/V data may have raggedness in terms of lengths,
   * we represent the the append lengths in CSR format.
   */
  NDArray cur_append_length_indptr_device_;
  /*! \brief The k position offset of applying RoPE for each sequence. */
  NDArray k_ragged_rope_pos_offset_device_;
  /*! \brief The q position mapping of applying RoPE for each sequence. */
  NDArray q_rope_position_map_device_;
  /*!
   * \brief The corresponding position in global KV cache (pages)
   * for each position along the length dimension of K/V data when
   * appending new K/V data.
   */
  NDArray append_position_map_device_;

  // Temporary arrays to store intermediate attention results.
  NDArray temp_attn_q_device_;
  NDArray temp_attn_k_device_;
  NDArray temp_attn_v_device_;
  NDArray temp_attn_output_device_;
  NDArray temp_attn_scores_device_;
  NDArray merged_attn_scores_device_;
  std::vector<NDArray> temp_attn_workspace_;

  //-------------------------------------------
  // Below are the auxiliary data structure on CPU.
  // We make them class members to avoid repetitive allocation time in BeginForward.
  //-------------------------------------------
  std::vector<std::vector<int32_t>> qo_indptr_on_depths_host_;
  std::vector<std::vector<int32_t>> page_indptr_on_depths_host_;
  std::vector<std::vector<int32_t>> page_indices_on_depths_host_;
  std::vector<std::vector<int32_t>> last_page_len_on_depths_host_;
  std::vector<std::vector<int32_t>> sliding_window_offset_on_depths_host_;
  std::vector<std::vector<int32_t>> sink_size_on_depths_host_;
  std::vector<std::vector<int32_t>> k_rope_pos_offset_on_depths_host_;
  std::vector<int32_t> k_ragged_rope_pos_offset_host_;
  std::vector<int32_t> q_rope_position_map_host_;
  std::vector<int32_t> append_position_map_host_;
  std::vector<int32_t> cur_append_lengths_indptr_host_;

  //-------------------------------------------
  // For efficient memory management, the actual sizes of the arrays
  // above are over allocated.
  // We create a view for the actual shapes of each of the arrays
  // after each synchronization and pass these views as input for
  // attention/append.
  //-------------------------------------------
  NDArray cur_append_length_indptr_view_;
  NDArray k_ragged_rope_pos_offset_view_;
  NDArray q_rope_position_map_view_;
  NDArray append_position_map_view_;
  NDArray temp_attn_output_view_;
  NDArray temp_attn_scores_view_;
  NDArray merged_attn_scores_view_;
  std::vector<NDArray> qo_indptr_on_depths_view_;
  std::vector<NDArray> page_indptr_on_depths_view_;
  std::vector<NDArray> page_indices_on_depths_view_;
  std::vector<NDArray> length_info_on_depths_view_;
  std::vector<NDArray> k_rope_pos_offset_view_;

  PackedFunc f_transpose_append_;
  PackedFunc f_attention_prefill_;
  PackedFunc f_attention_decode_;
  PackedFunc f_attention_prefill_sliding_window_;
  PackedFunc f_attention_decode_sliding_window_;
  PackedFunc f_attention_prefill_ragged_;
  Optional<PackedFunc> f_attention_prefill_ragged_begin_forward_;
  Optional<PackedFunc> f_attention_prefill_ragged_end_forward_;
  Optional<PackedFunc> f_attention_prefill_begin_forward_;
  Optional<PackedFunc> f_attention_prefill_end_forward_;
  Optional<PackedFunc> f_attention_decode_begin_forward_;
  Optional<PackedFunc> f_attention_decode_end_forward_;
  PackedFunc f_merge_inplace_;
  PackedFunc f_split_rotary_;
  Optional<PackedFunc> f_debug_get_kv_;

  /*! \brief Number of fork depth in the current round of forward. */
  int num_depths_;
  /*! \brief Whether to compute attention after appending KV into cache or not. */
  bool append_before_attn_;
  /*! \brief Whether to use decode kernel for each depth. (see GetChunkedBlockIds) */
  std::vector<bool> use_decode_kernel_;
  /*! \brief Whether the attention request is a decode request, set in BeginForwardFunction. */
  bool is_decode_request_;
  /*! \brief The device this PagedKVCache runs on. */
  DLDevice device_;
  /*! \brief The device stream for the default computation operations. */
  TVMStreamHandle compute_stream_ = nullptr;
  /*! \brief The device stream for copying auxiliary data structure to GPU. */
  TVMStreamHandle copy_stream_ = nullptr;

 public:
  /*! \brief Constructor. Take the cache configuration and initialize the NDArrays. */
  explicit PagedAttentionKVCacheObj(
      int64_t page_size,  //
      int64_t num_layers, int64_t num_qo_heads, int64_t num_kv_heads, int64_t head_dim,
      int64_t reserved_num_seqs, int64_t num_total_pages, int64_t prefill_chunk_size,
      bool support_sliding_window, RoPEMode rope_mode, double rotary_scale, double rotary_theta,
      DLDataType dtype, DLDevice device, PackedFunc f_transpose_append,
      PackedFunc f_attention_prefill, PackedFunc f_attention_decode,
      PackedFunc f_attention_prefill_sliding_window, PackedFunc f_attention_decode_sliding_window,
      PackedFunc f_attention_prefill_ragged,
      Optional<PackedFunc> f_attention_prefill_ragged_begin_forward,
      Optional<PackedFunc> f_attention_prefill_ragged_end_forward,
      Optional<PackedFunc> f_attention_prefill_begin_forward,
      Optional<PackedFunc> f_attention_prefill_end_forward,
      Optional<PackedFunc> f_attention_decode_begin_forward,
      Optional<PackedFunc> f_attention_decode_end_forward, PackedFunc f_merge_inplace,
      PackedFunc f_split_rotary, Optional<PackedFunc> f_debug_get_kv)
      : page_size_(page_size),
        num_layers_(num_layers),
        num_qo_heads_(num_qo_heads),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        num_total_pages_(num_total_pages),
        prefill_chunk_size_(prefill_chunk_size),
        support_sliding_window_(support_sliding_window),
        rope_mode_(support_sliding_window && rope_mode != RoPEMode::kNone ? RoPEMode::kInline
                                                                          : rope_mode),
        rotary_scale_(rotary_scale),
        rotary_theta_(rotary_theta),
        f_transpose_append_(std::move(f_transpose_append)),
        f_attention_prefill_(std::move(f_attention_prefill)),
        f_attention_decode_(std::move(f_attention_decode)),
        f_attention_prefill_sliding_window_(std::move(f_attention_prefill_sliding_window)),
        f_attention_decode_sliding_window_(std::move(f_attention_decode_sliding_window)),
        f_attention_prefill_ragged_(std::move(f_attention_prefill_ragged)),
        f_attention_prefill_ragged_begin_forward_(
            std::move(f_attention_prefill_ragged_begin_forward)),
        f_attention_prefill_ragged_end_forward_(std::move(f_attention_prefill_ragged_end_forward)),
        f_attention_prefill_begin_forward_(std::move(f_attention_prefill_begin_forward)),
        f_attention_prefill_end_forward_(std::move(f_attention_prefill_end_forward)),
        f_attention_decode_begin_forward_(std::move(f_attention_decode_begin_forward)),
        f_attention_decode_end_forward_(std::move(f_attention_decode_end_forward)),
        f_merge_inplace_(std::move(f_merge_inplace)),
        f_split_rotary_(std::move(f_split_rotary)),
        f_debug_get_kv_(std::move(f_debug_get_kv)),
        device_(device) {
    pages_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
      pages_.push_back(
          NDArray::Empty({num_total_pages, 2, num_kv_heads, page_size, head_dim}, dtype, device));
    }
    for (int d = 0; d < kPagedKVCacheMaxBlockDepth; ++d) {
      qo_indptr_on_depths_device_.push_back(
          NDArray::Empty({reserved_num_seqs + 1}, dtype_aux_, device));
      page_indptr_on_depths_device_.push_back(
          NDArray::Empty({reserved_num_seqs + 1}, dtype_aux_, device));
      page_indices_on_depths_device_.push_back(
          NDArray::Empty({num_total_pages}, dtype_aux_, device));
      length_info_on_depths_device_.push_back(
          NDArray::Empty({3, reserved_num_seqs}, dtype_aux_, device));
      k_rope_pos_offset_device_.push_back(NDArray::Empty({reserved_num_seqs}, dtype_aux_, device));
      temp_attn_workspace_.push_back(
          NDArray::Empty({kAttnWorkspaceByte / 4}, DataType::Float(32), device));
      qo_indptr_on_depths_view_.push_back(NDArray());
      page_indptr_on_depths_view_.push_back(NDArray());
      page_indices_on_depths_view_.push_back(NDArray());
      length_info_on_depths_view_.push_back(NDArray());
      k_rope_pos_offset_view_.push_back(NDArray());
    }
    // Additional workspace for the "prefill with ragged kv" kernel.
    temp_attn_workspace_.push_back(
        NDArray::Empty({kAttnWorkspaceByte / 4}, DataType::Float(32), device));
    cur_append_length_indptr_device_ = NDArray::Empty({reserved_num_seqs + 1}, dtype_aux_, device);
    k_ragged_rope_pos_offset_device_ = NDArray::Empty({reserved_num_seqs}, dtype_aux_, device);
    q_rope_position_map_device_ = NDArray::Empty({prefill_chunk_size_}, dtype_aux_, device);
    append_position_map_device_ = NDArray::Empty({prefill_chunk_size_}, dtype_aux_, device);

    temp_attn_q_device_ =
        NDArray::Empty({prefill_chunk_size_, num_qo_heads, head_dim}, dtype, device);
    temp_attn_k_device_ =
        NDArray::Empty({prefill_chunk_size_, num_kv_heads, head_dim}, dtype, device);
    temp_attn_v_device_ =
        NDArray::Empty({prefill_chunk_size_, num_kv_heads, head_dim}, dtype, device);
    temp_attn_output_device_ =
        NDArray::Empty({prefill_chunk_size_, num_qo_heads, head_dim}, dtype, device);
    temp_attn_scores_device_ =
        NDArray::Empty({prefill_chunk_size_, num_qo_heads}, DataType::Float(32), device);
    merged_attn_scores_device_ =
        NDArray::Empty({prefill_chunk_size_, num_qo_heads}, DataType::Float(32), device);
    for (int64_t page_id = num_total_pages - 1; page_id >= 0; --page_id) {
      free_page_ids_.push_back(page_id);
    }

    // If the device is CUDA/ROCm, we create a standalone copy stream, in
    // purpose to hide the latency of auxiliary stream copy.
    if (device.device_type == DLDeviceType::kDLCUDA ||
        device.device_type == DLDeviceType::kDLROCM) {
      // The compute stream is the default stream.
      compute_stream_ = DeviceAPI::Get(device)->GetCurrentStream(device);
      copy_stream_ = DeviceAPI::Get(device)->CreateStream(device);
    }
  }

  ~PagedAttentionKVCacheObj() {
    // Free the copy stream if defined.
    if (copy_stream_ != nullptr) {
      DeviceAPI::Get(device_)->FreeStream(device_, copy_stream_);
    }
  }

  /*! \brief Reset the KV cache. */
  void Clear() final {
    seq_map_.clear();
    ICHECK(pages_.defined());
    free_page_ids_.clear();
    for (int64_t page_id = num_total_pages_ - 1; page_id >= 0; --page_id) {
      free_page_ids_.push_back(page_id);
    }
    global_block_pool_.clear();
    free_block_idx_.clear();
    dirty_aux_data_device_ = false;
  }

  /************** Sequence Management **************/

  void AddSequence(int64_t seq_id) final {
    CHECK(seq_map_.find(seq_id) == seq_map_.end())
        << "The sequence \"" << seq_id << "\" is already in the KV cache.";
    int32_t block_idx = GetFreeBlock();
    seq_map_.insert({seq_id, Sequence(global_block_pool_, block_idx)});
    dirty_aux_data_device_ = true;
  }

  void RemoveSequence(int64_t seq_id) final {
    auto it = seq_map_.find(seq_id);
    CHECK(it != seq_map_.end()) << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";
    const Block& block = global_block_pool_[it->second.last_block_idx];
    CHECK_EQ(block.external_ref_cnt, 0)
        << "The sequence is currently referenced by other sequence and thus cannot be removed.";

    // - Decrease the external reference of the parent block.
    if (block.parent_idx != -1) {
      Block& parent_block = global_block_pool_[block.parent_idx];
      ICHECK_GT(parent_block.external_ref_cnt, 0);
      --parent_block.external_ref_cnt;
    }
    // - Free pages in the last block.
    for (int32_t page_id : block.page_ids) {
      free_page_ids_.push_back(page_id);
    }
    // - Remove the sequence from seq_map.
    free_block_idx_.push_back(it->second.last_block_idx);
    seq_map_.erase(it);
    dirty_aux_data_device_ = true;
  }

  void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id) final {
    auto parent_it = seq_map_.find(parent_seq_id);
    CHECK(parent_it != seq_map_.end())
        << "The parent sequence \"" << parent_seq_id << "\" cannot be found in KV cache.";
    CHECK(seq_map_.find(child_seq_id) == seq_map_.end())
        << "The child sequence \"" << child_seq_id << "\" is already in the KV cache.";
    CHECK_EQ(parent_it->second.sliding_window_size, -1)
        << "The parent sequence \"" << parent_seq_id
        << "\" is enabled with sliding window and thus cannot be forked.";

    int32_t parent_block_idx = parent_it->second.last_block_idx;
    ++global_block_pool_[parent_block_idx].external_ref_cnt;
    // Create a child block with the parent block pointer.
    int32_t child_block_idx = GetFreeBlock();
    global_block_pool_[child_block_idx].start_pos = parent_it->second.seq_length;
    global_block_pool_[child_block_idx].parent_idx = parent_block_idx;
    // Create the child sequence with the child block.
    seq_map_.insert({child_seq_id, Sequence(global_block_pool_, child_block_idx)});
    dirty_aux_data_device_ = true;
  }

  void EnableSlidingWindowForSeq(int64_t seq_id, int32_t sliding_window_size,
                                 int32_t attn_sink_size) final {
    CHECK(support_sliding_window_) << "The KV cache does not support sliding window.";
    auto it = seq_map_.find(seq_id);
    CHECK(it != seq_map_.end()) << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";
    CHECK_GE(attn_sink_size, 0)
        << "The specified attention sink size is expected to be non negative";
    CHECK_GT(sliding_window_size, 0) << "The specified sliding window size should be positive.";
    CHECK_LT(attn_sink_size, sliding_window_size)
        << "The attn sink size should be less than the sliding window size.";

    // Set the sliding window flag of the sequence.
    CHECK_EQ(it->second.sliding_window_size, -1)
        << "A sequence cannot be enabled twice for sliding window.";

    // Compute the total length of the prefix blocks of this sequence.
    Block& last_block = global_block_pool_[it->second.last_block_idx];
    int32_t prefix_length = it->second.seq_length - last_block.seq_length;
    ICHECK_GE(prefix_length, 0);
    // Since the prefix blocks cannot sliding, they are natural
    // attention sinks here. When the prefix length is already
    // larger than the specified attn sink size, we do not want to
    // introduce more sink. Therefore, we update the given attn sink size.
    it->second.last_block_attn_sink_size = std::max(attn_sink_size - prefix_length, 0);
    it->second.sliding_window_size = sliding_window_size;
  }

  void PopN(int64_t seq_id, int32_t n) final {
    auto it = seq_map_.find(seq_id);
    CHECK(it != seq_map_.end()) << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";

    Block& block = global_block_pool_[it->second.last_block_idx];
    CHECK_GE(n, 0) << "The length of popping " << n << " cannot be negative.";
    CHECK_LT(n, block.seq_length) << "The sequence only has length " << block.seq_length
                                  << " in the last block, while the length of pop is " << n
                                  << " which exceeds the last-block sequence length.";

    int64_t cur_npage = block.page_ids.size();
    int64_t tgt_npage = (block.seq_length - n + page_size_ - 1) / page_size_;
    while (cur_npage > tgt_npage) {
      free_page_ids_.push_back(block.page_ids.back());
      block.page_ids.pop_back();
      --cur_npage;
    }
    it->second.seq_length -= n;
    block.seq_length -= n;
    dirty_aux_data_device_ = true;
  }

  /************** Raw Info Query **************/

  int32_t GetNumAvailablePages() const final { return free_page_ids_.size(); }

  int32_t GetTotalSequenceLength() const final {
    int32_t total_seq_len = 0;
    for (const auto& it : seq_map_) {
      total_seq_len += it.second.seq_length;
    }
    return total_seq_len;
  }

  /************** Attention **************/

  void BeginForward(const IntTuple& seq_ids, const IntTuple& append_lengths) final {
    CHECK_EQ(seq_ids.size(), append_lengths.size())
        << "The seq_ids size (" << seq_ids.size() << ") and append_lengths size ("
        << append_lengths.size() << ") mismatch.";
    cur_batch_size_ = seq_ids.size();
    cur_append_lengths_ = append_lengths;

    // - Collect sequence/block/page information for attention.
    std::vector<Sequence*> sequences;
    std::vector<int32_t> last_block_length_before_append;
    is_decode_request_ = true;
    sequences.reserve(cur_batch_size_);
    last_block_length_before_append.reserve(cur_batch_size_);
    k_ragged_rope_pos_offset_host_.resize(cur_batch_size_);
    for (int i = 0; i < cur_batch_size_; ++i) {
      auto it = seq_map_.find(seq_ids[i]);
      CHECK(it != seq_map_.end()) << "The sequence \"" << seq_ids[i]
                                  << "\" cannot be found in KV cache.";
      sequences.push_back(&it->second);
      last_block_length_before_append.push_back(
          global_block_pool_[it->second.last_block_idx].seq_length);
      k_ragged_rope_pos_offset_host_[i] = it->second.seq_length;
      it->second.seq_length += append_lengths[i];
      if (append_lengths[i] != 1) {
        is_decode_request_ = false;
      }
    }

    std::vector<std::vector<int32_t>> block_ids_on_depths = GetBlockIdsOnDepth(sequences);
    num_depths_ = block_ids_on_depths.size();
    ICHECK_LE(num_depths_, kPagedKVCacheMaxBlockDepth);

    std::vector<std::vector<std::pair<int32_t, int32_t>>> chunked_block_ids_arr;
    chunked_block_ids_arr.reserve(num_depths_);
    use_decode_kernel_.clear();
    for (int d = 0; d < num_depths_; ++d) {
      auto [chunked_block_ids, use_decode_kernel] = GetChunkedBlockIds(block_ids_on_depths[d]);
      chunked_block_ids_arr.push_back(chunked_block_ids);
      use_decode_kernel_.push_back(use_decode_kernel);
    }

    append_before_attn_ = !support_sliding_window_ && num_depths_ == 1 && use_decode_kernel_[0];
    if (append_before_attn_) {
      // Right now we use different kernels when depth is 1 or not 1.
      // For the case where maximum depth is 1, we create the auxiliary
      // data structure with regard to the page table after appending.
      for (int i = 0; i < cur_batch_size_; ++i) {
        ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);
      }
    }

    qo_indptr_on_depths_host_.resize(num_depths_);
    page_indptr_on_depths_host_.resize(num_depths_);
    page_indices_on_depths_host_.resize(num_depths_);
    last_page_len_on_depths_host_.resize(num_depths_);
    sliding_window_offset_on_depths_host_.resize(num_depths_);
    sink_size_on_depths_host_.resize(num_depths_);
    k_rope_pos_offset_on_depths_host_.resize(num_depths_);

    for (int d = 0; d < num_depths_; ++d) {
      std::vector<int32_t>& qo_indptr_h = qo_indptr_on_depths_host_[d];
      std::vector<int32_t>& page_indptr_h = page_indptr_on_depths_host_[d];
      std::vector<int32_t>& page_indices_h = page_indices_on_depths_host_[d];
      std::vector<int32_t>& last_page_len_h = last_page_len_on_depths_host_[d];
      std::vector<int32_t>& sliding_window_offset_h = sliding_window_offset_on_depths_host_[d];
      std::vector<int32_t>& sink_size_h = sink_size_on_depths_host_[d];
      std::vector<int32_t>& k_rope_pos_offset_h = k_rope_pos_offset_on_depths_host_[d];
      qo_indptr_h.clear();
      page_indptr_h.clear();
      page_indices_h.clear();
      last_page_len_h.clear();
      sliding_window_offset_h.clear();
      sink_size_h.clear();
      k_rope_pos_offset_h.clear();
      qo_indptr_h.push_back(0);
      page_indptr_h.push_back(0);
      for (const auto& [block_id, chunk_append_length] : chunked_block_ids_arr[d]) {
        qo_indptr_h.push_back(qo_indptr_h.back() + chunk_append_length);
        if (block_id == -1) {
          page_indptr_h.push_back(page_indptr_h.back());
          last_page_len_h.push_back(0);
          sliding_window_offset_h.push_back(0);
          sink_size_h.push_back(0);
          k_rope_pos_offset_h.push_back(0);
        } else {
          const Block& block = global_block_pool_[block_id];
          page_indptr_h.push_back(page_indptr_h.back() + block.page_ids.size());
          page_indices_h.insert(page_indices_h.end(), block.page_ids.begin(), block.page_ids.end());
          last_page_len_h.push_back(block.seq_length == 0 ? 0
                                                          : (block.seq_length - block.sink_length +
                                                             block.sliding_window_offset - 1) %
                                                                    page_size_ +
                                                                1);
          sliding_window_offset_h.push_back(block.sliding_window_offset);
          sink_size_h.push_back(block.sink_length);
          k_rope_pos_offset_h.push_back(block.start_pos);
        }
      }
    }

    if (!append_before_attn_) {
      // Right now we use different kernels when depth is 1 or not 1.
      // For the case where maximum depth is not 1, we create the auxiliary
      // data structure with regard to the page table before appending.
      for (int i = 0; i < cur_batch_size_; ++i) {
        ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);
      }
    }

    // Map each the token position in the input batch to the position
    // in the global KV cache. The mapping is used in when appending k/v values.
    q_rope_position_map_host_.clear();
    append_position_map_host_.clear();
    for (int i = 0; i < cur_batch_size_; ++i) {
      int64_t append_length = append_lengths[i];
      const Block& block = global_block_pool_[sequences[i]->last_block_idx];
      for (int64_t pos = 0; pos < append_length; ++pos) {
        q_rope_position_map_host_.push_back(k_ragged_rope_pos_offset_host_[i] + pos);

        int32_t pos_in_block = block.seq_length - append_length + pos;
        if (last_block_length_before_append[i] + pos < block.sink_length) {
          // The location to write is part of the attention sink.
          int32_t offset_in_block = last_block_length_before_append[i] + pos;
          append_position_map_host_.push_back(block.page_ids[offset_in_block / page_size_] *
                                                  page_size_ +
                                              offset_in_block % page_size_);
        } else if (pos_in_block < block.sink_length) {
          // The location to write is pinned by attn sink before the append.
          // Therefore we cannot write into the location.
          append_position_map_host_.push_back(-1);
        } else {
          // The location to write is in the sliding window.
          int32_t offset_in_block = pos_in_block - block.sink_length + block.sliding_window_offset;
          append_position_map_host_.push_back(block.page_ids[offset_in_block / page_size_] *
                                                  page_size_ +
                                              offset_in_block % page_size_);
        }
      }
    }
  }

  void EndForward() final {
    if (!f_attention_prefill_end_forward_.defined() || !f_attention_decode_end_forward_.defined() ||
        !f_attention_prefill_ragged_end_forward_.defined()) {
      return;
    }
    f_attention_prefill_ragged_end_forward_.value()();
    for (int d = 0; d < num_depths_; ++d) {
      f_attention_prefill_end_forward_.value()(d);
      f_attention_decode_end_forward_.value()(d);
    }
  }

  void AttentionWithFusedQKV(int64_t layer_id, NDArray qkv_data, Optional<NDArray> mask,
                             NDArray o_data, double attn_score_scaling_factor) final {
    // Part 1. Shape and dtype check.
    NDArray pages = pages_[layer_id];
    CHECK(qkv_data.DataType() == pages.DataType());
    CHECK(o_data.DataType() == pages.DataType());

    // qkv_data: (num_total_length, num_qo_heads + 2 * num_kv_heads, head_dim)
    // o_data: (num_total_length, num_qo_heads, head_dim)

    CHECK_EQ(qkv_data->ndim, 3);
    CHECK_EQ(o_data->ndim, 3);
    for (int dim = 0; dim < 3; ++dim) {
      if (dim == 1) {
        CHECK_EQ(qkv_data->shape[1], num_qo_heads_ + 2 * num_kv_heads_);
        CHECK_EQ(o_data->shape[1], num_qo_heads_);
      } else {
        CHECK_EQ(o_data->shape[dim], qkv_data->shape[dim]);
      }
    }

    CHECK_EQ(qkv_data->shape[2], head_dim_);
    int64_t total_seq_length = 0;
    for (int64_t seq_id = 0; seq_id < cur_batch_size_; ++seq_id) {
      total_seq_length += cur_append_lengths_[seq_id];
    }
    CHECK_EQ(total_seq_length, qkv_data->shape[0]);
    // Sync the copy stream and the compute stream.
    ComputeStreamWaitForCopyStream();
    // The auxiliary data structure on device must have been synchronized.
    ICHECK(!dirty_aux_data_device_);

    NDArray q_data = temp_attn_q_device_.CreateView({total_seq_length, num_qo_heads_, head_dim_},
                                                    qkv_data->dtype);
    NDArray k_data = temp_attn_k_device_.CreateView({total_seq_length, num_kv_heads_, head_dim_},
                                                    qkv_data->dtype);
    NDArray v_data = temp_attn_v_device_.CreateView({total_seq_length, num_kv_heads_, head_dim_},
                                                    qkv_data->dtype);
    // Part 2. Split fused qkv and apply rotary embedding to q/k data.
    f_split_rotary_(qkv_data, q_rope_position_map_view_, q_data, k_data, v_data,
                    rope_mode_ == RoPEMode::kNormal);

    // Part 3. Append k/v data to kv-cache if flag "append_before_attn" is set.
    if (append_before_attn_) {
      f_transpose_append_(pages_[layer_id], k_data, v_data, append_position_map_view_);
    }
    // Part 4: perform attention
    AttentionInternal(layer_id, q_data, k_data, v_data, o_data, attn_score_scaling_factor);
    // Part 5. Append k/v data to kv-cache if flag "append_before_attn" is not set.
    if (!append_before_attn_) {
      f_transpose_append_(pages_[layer_id], k_data, v_data, append_position_map_view_);
    }
  }

  NDArray GetQueryPositions() final {
    // Sync the copy stream and the compute stream.
    ComputeStreamWaitForCopyStream();
    // The auxiliary data structure on device must have been synchronized.
    ICHECK(!dirty_aux_data_device_);
    return q_rope_position_map_view_;
  };

  void DebugGetKV(int64_t seq_id, int64_t start_pos, int64_t end_pos, NDArray k_data,
                  NDArray v_data) final {
    CHECK(f_debug_get_kv_.defined())
        << "PageAttentionKVCache requires the `f_debug_get_kv` to be explicitly passed in when "
           "initialization. Please construct the KV cache with `f_debug_get_kv`.";

    const Sequence& seq = seq_map_.at(seq_id);
    CHECK_GE(start_pos, 0) << "DebugGetKV does not accept negative start_pos " << start_pos;
    CHECK_LE(end_pos, seq.seq_length) << "DebugGetKV does not accept out-of-range end_pos";
    CHECK_LT(start_pos, end_pos) << "DebugGetKV does not accept \"start_pos >= end_pos\"";

    // k/v_data: (num_layers, seq_length, num_kv_heads, head_dim)
    static constexpr const char* error_msg =
        "DebugGetKV expects the k_data in layout (num_layers, seq_length, num_kv_heads, head_dim).";
    std::vector<NDArray*> vec_kv_data = {&k_data, &v_data};
    for (const NDArray* data_ptr : vec_kv_data) {
      CHECK_EQ((*data_ptr)->ndim, 4) << error_msg;
      CHECK_EQ((*data_ptr)->shape[0], num_layers_)
          << error_msg << " The number of layers mismatches.";
      CHECK_EQ((*data_ptr)->shape[1], end_pos - start_pos)
          << error_msg << " The sequence length mismatches.";
      CHECK_EQ((*data_ptr)->shape[2], num_kv_heads_)
          << error_msg << " The number of heads mismatches.";
      CHECK_EQ((*data_ptr)->shape[3], head_dim_)
          << error_msg << " The number of head features mismatches.";
    }

    std::vector<int32_t> trace = seq.GetBlockTrace(global_block_pool_);
    std::vector<int32_t> append_position_map;
    append_position_map.reserve(seq.seq_length);
    for (int32_t block_id : trace) {
      const Block& block = global_block_pool_[block_id];
      for (int i = 0; i < block.seq_length; ++i) {
        int32_t offset =
            i < block.sink_length ? i : i - block.sink_length + block.sliding_window_offset;
        int page_id = block.page_ids[offset / page_size_];
        int page_offset = offset % page_size_;
        append_position_map.push_back(page_id * page_size_ + page_offset);
      }
    }
    NDArray position_map_device =
        NDArray::Empty({end_pos - start_pos}, dtype_aux_, cur_append_length_indptr_device_->device);
    position_map_device.CopyFromBytes(
        append_position_map.data() + start_pos,
        (end_pos - start_pos) * ((dtype_aux_.bits * dtype_aux_.lanes + 7) / 8));
    for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
      f_debug_get_kv_.value()(pages_[layer_id], position_map_device, k_data, v_data, layer_id);
    }
  }

  void DebugSetKV(int64_t seq_id, int64_t start_pos, NDArray k_data, NDArray v_data) final {
    ICHECK(false) << "DebugSetKV for PageAttentionKVCache not implemented yet.";
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.vm.PagedAttentionKVCache";
  TVM_DECLARE_FINAL_OBJECT_INFO(PagedAttentionKVCacheObj, AttentionKVCacheObj);

 private:
  /*! \brief Get a new free page and return its id. */
  int32_t GetFreePage() {
    // Find a page from the free page pools.
    CHECK(!free_page_ids_.empty()) << "The KV cache is full. No page can be allocated.";
    int32_t page_id = free_page_ids_.back();
    free_page_ids_.pop_back();
    return page_id;
  }

  /*! \brief Get a new free block and return its index. */
  int32_t GetFreeBlock() {
    if (!free_block_idx_.empty()) {
      int32_t block_idx = free_block_idx_.back();
      free_block_idx_.pop_back();
      global_block_pool_[block_idx].Reset();
      ICHECK_EQ(global_block_pool_[block_idx].index, block_idx);
      return block_idx;
    }

    int32_t block_idx = global_block_pool_.size();
    global_block_pool_.push_back(Block(block_idx));
    return block_idx;
  }

  /*!
   * \brief Slide the KV cache window of the given sequence when
   * it has sliding window enabled.
   * \param seq The sequence to be slidden when
   */
  void SlideWindowForSequence(Sequence* seq) {
    // - No action when the sequence is not enabled for sliding window.
    if (seq->sliding_window_size == -1) {
      return;
    }
    // - No action when the sequence length does not exceed the window size.
    if (seq->seq_length <= seq->sliding_window_size) {
      return;
    }

    int32_t length_to_slide = seq->seq_length - seq->sliding_window_size;
    // - Get the last block of the sequence.
    Block& block = global_block_pool_[seq->last_block_idx];

    // - If the attention sink exists and the last block has no previous
    // sink length, it means this is the first time we slide the sequence,
    // and thus we set the sink length of the last block, the index of the
    // first sliding page, and starting offset in first sliding page.
    if (seq->last_block_attn_sink_size > 0 && block.sink_length == 0) {
      ICHECK_EQ(block.sliding_window_offset, 0);
      block.sink_length = seq->last_block_attn_sink_size;
      block.sliding_window_offset = seq->last_block_attn_sink_size;
    }

    // - The sink pages cannot be slidden.
    int32_t num_sink_pages = (block.sink_length + page_size_ - 1) / page_size_;

    // - Compute the first sliding page index and in-page sliding window
    // start offset in the first sliding page after sliding.
    int32_t page_idx_after_sliding = (block.sliding_window_offset + length_to_slide) / page_size_;
    int32_t page_start_offset_after_sliding =
        (block.sliding_window_offset + length_to_slide) % page_size_;

    // - Free the pages that are fully slidden.
    while (page_idx_after_sliding > num_sink_pages) {
      if (block.page_ids[num_sink_pages] != kPagedKVCacheTempPageId) {
        free_page_ids_.push_back(block.page_ids[num_sink_pages]);
      }
      block.page_ids.erase(block.page_ids.begin() + num_sink_pages);
      --page_idx_after_sliding;
    }
    // - The first sliding page after sliding is either the last sink page,
    // or the page next to the last sink page.
    ICHECK(page_idx_after_sliding == num_sink_pages - 1 ||
           page_idx_after_sliding == num_sink_pages);

    // - Update the length of the sequence and the block.
    seq->seq_length = seq->sliding_window_size;
    block.seq_length -= length_to_slide;
    block.sliding_window_offset =
        page_idx_after_sliding * page_size_ + page_start_offset_after_sliding;
    ICHECK_GE(block.seq_length, block.sink_length);
    ICHECK_GE(block.sliding_window_offset, block.sink_length);
    ICHECK_EQ(
        (block.sliding_window_offset + (block.seq_length - block.sink_length) + page_size_ - 1) /
            page_size_,
        block.page_ids.size());
  }

  /*!
   * \brief Reserve extra append length in the last block of the given
   * sequence, as preparation of the incoming KV cache append.
   * New pages will be allocated to the block until the total
   * capacity can cover the current sequence length (before reservation)
   * plus the required append length.
   * \param block_idx The index of the block to process.
   * \param append_length The extra append length to reserve for the block.
   * \note We apply sliding window in this function.
   */
  void ReserveAppendLengthInSeq(Sequence* seq, int64_t append_length) {
    int32_t block_idx = seq->last_block_idx;
    Block& block = global_block_pool_[block_idx];
    CHECK_GT(append_length, 0) << "Append with length 0 is not allowed.";
    CHECK_EQ(block.external_ref_cnt, 0)
        << "The block is " << block.external_ref_cnt
        << "-time referenced by other blocks, thus cannot accept new KV values.";

    // ==================== Reserve ====================
    // The reservation is based on the current sequence length.
    // If "current sequence + append length" does not exceed the
    // current capacity (number of pages * page size), no action is taken.
    int64_t cur_npage = block.page_ids.size();
    int64_t tgt_npage = (block.seq_length - block.sink_length + block.sliding_window_offset +
                         append_length + page_size_ - 1) /
                        page_size_;
    for (int64_t page_idx = cur_npage; page_idx < tgt_npage; ++page_idx) {
      // When sliding window is enabled for the seq, we can "borrow temporary pages (-1)",
      // since the pages need to be slidden out might not have been released.
      if (free_page_ids_.empty() && seq->sliding_window_size != -1) {
        block.page_ids.push_back(kPagedKVCacheTempPageId);
      } else {
        block.page_ids.push_back(GetFreePage());
      }
    }
    block.seq_length += append_length;

    // ==================== Slide ====================
    // Slide the sequences so that the pages exceed the sliding window are released.
    SlideWindowForSequence(seq);
    for (int i = 0; i < static_cast<int>(block.page_ids.size()); ++i) {
      if (block.page_ids[i] == kPagedKVCacheTempPageId) {
        // Re-allocate the temporary pages after sliding window release.
        block.page_ids[i] = GetFreePage();
      }
    }

    dirty_aux_data_device_ = true;
  }

  /*!
   * \brief For the given list of sequences, check the block trace of
   * each sequence, and return the blocks ids used by the sequences
   * on each depth.
   * More precisely, the inner returned vector contains the block ids
   * used by the sequences on a certain depth (or "-1" if a sequence
   * has fewer depth). The outer returned vector contains the inner
   * vectors from the lowest depth to the highest depth.
   */
  std::vector<std::vector<int32_t>> GetBlockIdsOnDepth(
      const std::vector<Sequence*>& sequences) const {
    // - Get the trace of each sequence.
    int64_t num_depths = 0;
    std::vector<std::vector<int32_t>> seq_block_traces;
    seq_block_traces.reserve(cur_batch_size_);
    for (int i = 0; i < cur_batch_size_; ++i) {
      std::vector<int32_t> trace = sequences[i]->GetBlockTrace(global_block_pool_);
      num_depths = std::max(num_depths, static_cast<int64_t>(trace.size()));
      seq_block_traces.push_back(std::move(trace));
    }

    // "Transpose" the traces, yielding the block ids used on each depth.
    std::vector<std::vector<int32_t>> block_ids_on_depths;
    block_ids_on_depths.reserve(num_depths);
    for (int d = 0; d < num_depths; ++d) {
      std::vector<int32_t> block_ids;
      block_ids.reserve(cur_batch_size_);
      for (int i = 0; i < cur_batch_size_; ++i) {
        block_ids.push_back(
            d < static_cast<int>(seq_block_traces[i].size()) ? seq_block_traces[i][d] : -1);
      }
      block_ids_on_depths.push_back(std::move(block_ids));
    }
    return block_ids_on_depths;
  }

  /*!
   * \brief This function considers an optimization which coalesces
   * adjacent decode attention computations into a single prefill
   * attention computation if the adjacent decodes attend to the same
   * k/v values under certain conditions.
   * If it decides to coalesce on a certain depth, we need to know
   * the prefill length after coalescing. This function returns
   * - a vector of block ids together with the prefill/decode lengths
   * that attend to the blocks.
   * - a boolean indicating whether to use decode kernel on for the
   * input blocks.
   */
  std::pair<std::vector<std::pair<int32_t, int32_t>>, bool> GetChunkedBlockIds(
      const std::vector<int32_t>& block_ids) const {
    std::vector<std::pair<int32_t, int32_t>> uncoalesced_block_ids;
    std::vector<std::pair<int32_t, int32_t>> coalesced_block_ids;

    // Gather the number of pages before/after coalescing respectively.
    int cur_block_id = block_ids[0];
    int chunk_append_length = cur_append_lengths_[0];
    int page_counter_coalesced = 0;
    int page_counter_uncoalesced =
        block_ids[0] != -1 ? global_block_pool_[block_ids[0]].page_ids.size() : 0;
    for (int i = 1; i < static_cast<int>(block_ids.size()); ++i) {
      if (block_ids[i] != -1) {
        page_counter_uncoalesced += global_block_pool_[block_ids[i]].page_ids.size();
      }
      uncoalesced_block_ids.emplace_back(block_ids[i - 1], cur_append_lengths_[i - 1]);
      if (block_ids[i] == cur_block_id) {
        chunk_append_length += cur_append_lengths_[i];
      } else {
        coalesced_block_ids.emplace_back(cur_block_id, chunk_append_length);
        if (cur_block_id != -1) {
          page_counter_coalesced += global_block_pool_[cur_block_id].page_ids.size();
        }
        cur_block_id = block_ids[i];
        chunk_append_length = cur_append_lengths_[i];
      }
    }
    uncoalesced_block_ids.emplace_back(block_ids.back(), cur_append_lengths_.back());
    coalesced_block_ids.emplace_back(cur_block_id, chunk_append_length);
    if (cur_block_id != -1) {
      page_counter_coalesced += global_block_pool_[cur_block_id].page_ids.size();
    }
    double coalesce_ratio = 1.0 * page_counter_uncoalesced / page_counter_coalesced;
    // Do not coalesce and use batch decode kernel when coalesce ratio is small.
    bool use_decode_kernel = is_decode_request_ && coalesce_ratio < 1.1;

    return {use_decode_kernel ? uncoalesced_block_ids : coalesced_block_ids, use_decode_kernel};
  }

  /*! \brief Invoke the "begin forward" functions of underlying kernels. */
  void KernelBeginForward() {
    if (!f_attention_prefill_begin_forward_.defined() ||
        !f_attention_decode_begin_forward_.defined() ||
        !f_attention_prefill_ragged_begin_forward_.defined()) {
      return;
    }

    if (append_before_attn_) {
      if (!support_sliding_window_) {
        f_attention_decode_begin_forward_.value()(
            /*depth=*/0, temp_attn_workspace_[1], page_indptr_on_depths_view_[0],
            length_info_on_depths_view_[0], num_qo_heads_, num_kv_heads_, head_dim_, page_size_,
            /*rotary_mode=*/rope_mode_ == RoPEMode::kInline, copy_stream_);
      }
    } else {
      f_attention_prefill_ragged_begin_forward_.value()(
          temp_attn_workspace_[0], cur_append_length_indptr_view_, cur_batch_size_, num_qo_heads_,
          num_kv_heads_, head_dim_, copy_stream_);
      if (support_sliding_window_) {
        return;
      }
      for (int d = 0; d < num_depths_; ++d) {
        if (page_indices_on_depths_view_[d]->shape[0] == 0) {
          continue;
        }
        if (use_decode_kernel_[d]) {
          f_attention_decode_begin_forward_.value()(
              d, temp_attn_workspace_[d + 1], page_indptr_on_depths_view_[d],
              length_info_on_depths_view_[d], num_qo_heads_, num_kv_heads_, head_dim_, page_size_,
              /*rotary_mode=*/rope_mode_ == RoPEMode::kInline, copy_stream_);
        } else {
          f_attention_prefill_begin_forward_.value()(
              /*depth=*/d, temp_attn_workspace_[d + 1], qo_indptr_on_depths_view_[d],
              length_info_on_depths_view_[d]->shape[0], num_qo_heads_, num_kv_heads_, head_dim_,
              copy_stream_);
        }
      }
    }
  }

  /*!
   * \brief Compute attention for between the input q data and the
   * input k/v data and the k/v data in cache on the given layer.
   */
  void AttentionInternal(int64_t layer_id, NDArray q_data, NDArray k_data, NDArray v_data,
                         NDArray output, double attn_score_scaling_factor) {
    PackedFunc f_prefill =
        !support_sliding_window_ ? f_attention_prefill_ : f_attention_prefill_sliding_window_;
    PackedFunc f_decode =
        !support_sliding_window_ ? f_attention_decode_ : f_attention_decode_sliding_window_;
    CHECK_GE(num_depths_, 1) << "The number of effective depths must be greater or equal to 1.";
    if (append_before_attn_) {
      f_decode(
          /*depth=*/0, q_data, pages_[layer_id], page_indptr_on_depths_view_[0],
          page_indices_on_depths_view_[0], length_info_on_depths_view_[0],
          k_rope_pos_offset_view_[0], q_rope_position_map_view_, output, merged_attn_scores_view_,
          /*rotary_mode=*/rope_mode_ == RoPEMode::kInline, rotary_scale_, rotary_theta_,
          attn_score_scaling_factor);
    } else {
      // Compute appended text self-attention
      f_attention_prefill_ragged_(q_data, cur_append_length_indptr_view_, k_data, v_data,
                                  cur_append_length_indptr_view_, q_rope_position_map_view_,
                                  k_ragged_rope_pos_offset_view_, output, merged_attn_scores_view_,
                                  /*causal=*/1,
                                  /*rotary_mode=*/rope_mode_ == RoPEMode::kInline, rotary_scale_,
                                  rotary_theta_, attn_score_scaling_factor);

      for (int d = 0; d < num_depths_; ++d) {
        if (page_indices_on_depths_view_[d]->shape[0] == 0) {
          continue;
        }
        if (use_decode_kernel_[d]) {
          // Use decode kernel for depth d
          f_decode(/*depth=*/d, q_data, pages_[layer_id], page_indptr_on_depths_view_[d],
                   page_indices_on_depths_view_[d], length_info_on_depths_view_[d],
                   k_rope_pos_offset_view_[d], q_rope_position_map_view_, temp_attn_output_view_,
                   temp_attn_scores_view_,
                   /*rotary_mode=*/rope_mode_ == RoPEMode::kInline, rotary_scale_, rotary_theta_,
                   attn_score_scaling_factor);
        } else {
          // Use prefill kernel for depth d
          f_prefill(
              /*depth=*/d, q_data, qo_indptr_on_depths_view_[d], pages_[layer_id],
              page_indptr_on_depths_view_[d], page_indices_on_depths_view_[d],
              length_info_on_depths_view_[d], k_rope_pos_offset_view_[d], q_rope_position_map_view_,
              temp_attn_output_view_, temp_attn_scores_view_,
              /*causal=*/0,
              /*rotary_mode=*/rope_mode_ == RoPEMode::kInline, rotary_scale_, rotary_theta_,
              attn_score_scaling_factor);
        }
        f_merge_inplace_(output, merged_attn_scores_view_, temp_attn_output_view_,
                         temp_attn_scores_view_);
      }
    }
  }

  /*! \brief Synchronize the copy stream and the compute stream. */
  void ComputeStreamWaitForCopyStream() {
    if (!dirty_aux_data_device_) {
      // If the auxiliary data is already synced, return and no need to sync again.
      return;
    }
    // - Sync NDArrays to GPU.
    SyncAuxArrayToDevice();
    KernelBeginForward();
    // - Clear the dirty flag.
    dirty_aux_data_device_ = false;
    // - If there is no particular copy stream, no action is needed.
    if (copy_stream_ == nullptr) {
      return;
    }
    // - Sync two streams.
    DeviceAPI::Get(device_)->SyncStreamFromTo(device_, copy_stream_, compute_stream_);
  }

  /*!
   * \brief Copy a vector of data to the input NDArray.
   * It optionally supports specifying the shape of copy and the element
   * offset to the destination NDArray.
   */
  void CopyVecDataToArray(NDArray array, int32_t* vec_data, Optional<ShapeTuple> shape = NullOpt,
                          int dst_elem_offset = 0) {
    DLTensor copy_dst = *array.operator->();
    if (shape.defined()) {
      ICHECK_EQ(shape.value().size(), 1);
      copy_dst.ndim = 1;
      copy_dst.shape = shape.value()->data;
    }
    copy_dst.byte_offset = dst_elem_offset * sizeof(int32_t);

    DLTensor copy_src;
    copy_src.data = vec_data;
    copy_src.device = Device{kDLCPU, 0};
    copy_src.ndim = 1;
    copy_src.dtype = array->dtype;
    copy_src.shape = copy_dst.shape;
    copy_src.strides = nullptr;
    copy_src.byte_offset = 0;
    NDArray::CopyFromTo(&copy_src, &copy_dst, copy_stream_);
  }

  /*!
   * \brief Synchronize auxiliary arrays to device.
   * \note This method resets the dirty flag to false, and needs to be
   * invoked before running attention computation on device.
   */
  void SyncAuxArrayToDevice() {
    ICHECK(dtype_aux_.bits == 32 && dtype_aux_.code == kDLInt);
    ICHECK_EQ(qo_indptr_on_depths_host_.size(), num_depths_);
    ICHECK_EQ(page_indptr_on_depths_host_.size(), num_depths_);
    ICHECK_EQ(page_indices_on_depths_host_.size(), num_depths_);
    ICHECK_EQ(last_page_len_on_depths_host_.size(), num_depths_);
    int64_t total_append_length = 0;
    int num_sequences = cur_append_lengths_.size();
    cur_append_lengths_indptr_host_.resize(num_sequences + 1);
    cur_append_lengths_indptr_host_[0] = 0;
    for (int i = 0; i < num_sequences; ++i) {
      cur_append_lengths_indptr_host_[i + 1] =
          cur_append_lengths_indptr_host_[i] + cur_append_lengths_[i];
    }
    total_append_length = cur_append_lengths_indptr_host_.back();
    ICHECK_EQ(total_append_length, append_position_map_host_.size());

    // 1. qo_indptr_on_depths
    for (int d = 0; d < num_depths_; ++d) {
      qo_indptr_on_depths_view_[d] = qo_indptr_on_depths_device_[d].CreateView(
          {static_cast<int64_t>(qo_indptr_on_depths_host_[d].size())}, dtype_aux_);
      CopyVecDataToArray(qo_indptr_on_depths_view_[d], qo_indptr_on_depths_host_[d].data());
    }

    // 2. page_indptr_on_depths
    for (int d = 0; d < num_depths_; ++d) {
      ICHECK_EQ(page_indptr_on_depths_host_[d].size(), qo_indptr_on_depths_host_[d].size());
      page_indptr_on_depths_view_[d] = page_indptr_on_depths_device_[d].CreateView(
          {static_cast<int64_t>(page_indptr_on_depths_host_[d].size())}, dtype_aux_);
      CopyVecDataToArray(page_indptr_on_depths_view_[d], page_indptr_on_depths_host_[d].data());
    }

    // 3. page_indices_on_depths
    for (int d = 0; d < num_depths_; ++d) {
      ICHECK_EQ(page_indices_on_depths_host_[d].size(), page_indptr_on_depths_host_[d].back());
      page_indices_on_depths_view_[d] = page_indices_on_depths_device_[d].CreateView(
          {static_cast<int64_t>(page_indices_on_depths_host_[d].size())}, dtype_aux_);
      if (!page_indices_on_depths_host_[d].empty()) {
        CopyVecDataToArray(page_indices_on_depths_view_[d], page_indices_on_depths_host_[d].data());
      }
    }

    // 4. length_info_on_depths
    // last_page_len_on_depths_host_;
    // sliding_window_offset_on_depths_host_;
    // sink_size_on_depths_host_;
    for (int d = 0; d < num_depths_; ++d) {
      int num_seq_on_layer = static_cast<int>(qo_indptr_on_depths_host_[d].size()) - 1;
      ICHECK_EQ(last_page_len_on_depths_host_[d].size(), num_seq_on_layer);
      ICHECK_EQ(sliding_window_offset_on_depths_host_[d].size(), num_seq_on_layer);
      ICHECK_EQ(sink_size_on_depths_host_[d].size(), num_seq_on_layer);
      if (!support_sliding_window_) {
        // Sliding window is not enabled, so we first copy "last_page_len".
        length_info_on_depths_view_[d] =
            length_info_on_depths_device_[d].CreateView({num_seq_on_layer}, dtype_aux_);
        CopyVecDataToArray(length_info_on_depths_view_[d], last_page_len_on_depths_host_[d].data());
      } else {
        // Sliding window is enabled,
        length_info_on_depths_view_[d] =
            length_info_on_depths_device_[d].CreateView({3, num_seq_on_layer}, dtype_aux_);
        ShapeTuple copy_shape{num_seq_on_layer};
        CopyVecDataToArray(length_info_on_depths_view_[d], last_page_len_on_depths_host_[d].data(),
                           copy_shape);
        CopyVecDataToArray(length_info_on_depths_view_[d],
                           sliding_window_offset_on_depths_host_[d].data(), copy_shape,
                           /*dst_elem_offset=*/num_seq_on_layer);
        CopyVecDataToArray(length_info_on_depths_view_[d], sink_size_on_depths_host_[d].data(),
                           copy_shape, /*dst_elem_offset=*/2 * num_seq_on_layer);
      }
    }

    // 5. k_rope_pos_offset_on_depths
    for (int d = 0; d < num_depths_; ++d) {
      ICHECK_EQ(k_rope_pos_offset_on_depths_host_[d].size() + 1,
                qo_indptr_on_depths_host_[d].size());
      k_rope_pos_offset_view_[d] = k_rope_pos_offset_device_[d].CreateView(
          {static_cast<int64_t>(k_rope_pos_offset_on_depths_host_[d].size())}, dtype_aux_);
      CopyVecDataToArray(k_rope_pos_offset_view_[d], k_rope_pos_offset_on_depths_host_[d].data());
    }

    // 6. cur_append_lengths_indptr
    cur_append_length_indptr_view_ =
        cur_append_length_indptr_device_.CreateView({num_sequences + 1}, dtype_aux_);
    CopyVecDataToArray(cur_append_length_indptr_view_, cur_append_lengths_indptr_host_.data());

    // 7. k_ragged_rope_pos_offset
    ICHECK_EQ(k_ragged_rope_pos_offset_host_.size(), num_sequences);
    k_ragged_rope_pos_offset_view_ =
        k_ragged_rope_pos_offset_device_.CreateView({num_sequences}, dtype_aux_);
    CopyVecDataToArray(k_ragged_rope_pos_offset_view_, k_ragged_rope_pos_offset_host_.data());

    // 8. q_rope_position_map
    ICHECK_EQ(q_rope_position_map_host_.size(), total_append_length);
    q_rope_position_map_view_ =
        q_rope_position_map_device_.CreateView({total_append_length}, dtype_aux_);
    CopyVecDataToArray(q_rope_position_map_view_, q_rope_position_map_host_.data());

    // 9. append_position_map
    append_position_map_view_ =
        append_position_map_device_.CreateView({total_append_length}, dtype_aux_);
    CopyVecDataToArray(append_position_map_view_, append_position_map_host_.data());

    // 10. Create view for temporary arrays for attention computation.
    temp_attn_output_view_ = temp_attn_output_device_.CreateView(
        {total_append_length, num_qo_heads_, head_dim_}, temp_attn_output_device_->dtype);
    temp_attn_scores_view_ = temp_attn_scores_device_.CreateView(
        {total_append_length, num_qo_heads_}, temp_attn_scores_device_->dtype);
    merged_attn_scores_view_ = merged_attn_scores_device_.CreateView(
        {total_append_length, num_qo_heads_}, merged_attn_scores_device_->dtype);

    // - Reset the dirty flag to false.
    dirty_aux_data_device_ = false;
  }
};

TVM_REGISTER_OBJECT_TYPE(PagedAttentionKVCacheObj);

//-------------------------------------------------
//  Register runtime functions
//-------------------------------------------------

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_create")
    .set_body_typed([](ShapeTuple cache_config, int64_t num_layers, int64_t num_qo_heads,
                       int64_t num_kv_heads, int64_t head_dim, int rope_mode, double rotary_scale,
                       double rotary_theta, NDArray init, PackedFunc f_transpose_append,
                       PackedFunc f_attention_prefill, PackedFunc f_attention_decode,
                       PackedFunc f_attention_prefill_sliding_window,  //
                       PackedFunc f_attention_decode_sliding_window,
                       PackedFunc f_attention_prefill_ragged,
                       PackedFunc f_attention_prefill_ragged_begin_forward,
                       PackedFunc f_attention_prefill_ragged_end_forward,
                       PackedFunc f_attention_prefill_begin_forward,
                       PackedFunc f_attention_prefill_end_forward,
                       PackedFunc f_attention_decode_begin_forward,
                       PackedFunc f_attention_decode_end_forward, PackedFunc f_merge_inplace,
                       PackedFunc f_split_rotary, Optional<PackedFunc> f_debug_get_kv) {
      CHECK_EQ(cache_config.size(), 5);
      int64_t reserved_num_seqs = cache_config[0];
      int64_t total_token_capacity = cache_config[1];
      int64_t prefill_chunk_size = cache_config[2];
      int64_t page_size = cache_config[3];
      bool support_sliding_window = cache_config[4];
      int64_t num_total_pages = (total_token_capacity + page_size - 1) / page_size;
      if (support_sliding_window) {
        // When sliding window is enabled, each sequence may use two more pages at most.
        num_total_pages += reserved_num_seqs * 2;
      }
      ObjectPtr<PagedAttentionKVCacheObj> n = make_object<PagedAttentionKVCacheObj>(
          page_size, num_layers, num_qo_heads, num_kv_heads, head_dim, reserved_num_seqs,
          num_total_pages, prefill_chunk_size, support_sliding_window, RoPEMode(rope_mode),
          rotary_scale, rotary_theta, init->dtype, init->device, std::move(f_transpose_append),
          std::move(f_attention_prefill), std::move(f_attention_decode),
          std::move(f_attention_prefill_sliding_window),
          std::move(f_attention_decode_sliding_window), std::move(f_attention_prefill_ragged),
          std::move(f_attention_prefill_ragged_begin_forward),
          std::move(f_attention_prefill_ragged_end_forward),
          std::move(f_attention_prefill_begin_forward), std::move(f_attention_prefill_end_forward),
          std::move(f_attention_decode_begin_forward), std::move(f_attention_decode_end_forward),
          std::move(f_merge_inplace), std::move(f_split_rotary), std::move(f_debug_get_kv));
      return AttentionKVCache(std::move(n));
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_create_reduced")
    .set_body_typed([](ShapeTuple cache_config, int64_t num_layers, int64_t num_qo_heads,
                       int64_t num_kv_heads, int64_t head_dim, int rope_mode, double rotary_scale,
                       double rotary_theta, NDArray init, PackedFunc f_transpose_append,
                       PackedFunc f_attention_prefill, PackedFunc f_attention_decode,
                       PackedFunc f_attention_prefill_sliding_window,
                       PackedFunc f_attention_decode_sliding_window,
                       PackedFunc f_attention_prefill_ragged, PackedFunc f_merge_inplace,
                       PackedFunc f_split_rotary, Optional<PackedFunc> f_debug_get_kv) {
      CHECK_EQ(cache_config.size(), 5);
      int64_t reserved_num_seqs = cache_config[0];
      int64_t total_token_capacity = cache_config[1];
      int64_t prefill_chunk_size = cache_config[2];
      int64_t page_size = cache_config[3];
      bool support_sliding_window = cache_config[4];
      int64_t num_total_pages = (total_token_capacity + page_size - 1) / page_size;
      if (support_sliding_window) {
        // When sliding window is enabled, each sequence may use two more pages at most.
        num_total_pages += reserved_num_seqs * 2;
      }
      ObjectPtr<PagedAttentionKVCacheObj> n = make_object<PagedAttentionKVCacheObj>(
          page_size, num_layers, num_qo_heads, num_kv_heads, head_dim, reserved_num_seqs,
          num_total_pages, prefill_chunk_size, support_sliding_window, RoPEMode(rope_mode),
          rotary_scale, rotary_theta, init->dtype, init->device, std::move(f_transpose_append),
          std::move(f_attention_prefill), std::move(f_attention_decode),
          std::move(f_attention_prefill_sliding_window),
          std::move(f_attention_decode_sliding_window), std::move(f_attention_prefill_ragged),  //
          NullOpt, NullOpt, NullOpt, NullOpt, NullOpt, NullOpt,                                 //
          std::move(f_merge_inplace), std::move(f_split_rotary), std::move(f_debug_get_kv));
      return AttentionKVCache(std::move(n));
    });

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
