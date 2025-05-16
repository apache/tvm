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
 * \file src/runtime/relax_vm/attn_utils.h
 * \brief Data structure and utilities for KV cache.
 */

#ifndef TVM_RUNTIME_RELAX_VM_ATTN_UTILS_H_
#define TVM_RUNTIME_RELAX_VM_ATTN_UTILS_H_

#include <tvm/runtime/ndarray.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>
#if defined(OPENCL_ENABLE_HOST_PTR)
#include "../opencl/opencl_common.h"
#endif

namespace tvm {
namespace runtime {
namespace relax_vm {

/*!
 * \brief The maximum allowed block depth (a.k.a. number of common
 * prefixes) in paged KV cache.
 */
constexpr const int kPagedKVCacheMaxBlockDepth = 2;
/*! \brief The maximum tree size of a single sequence in tree attention. */
constexpr const int kTreeAttnMaxTreeSize = 256;
/*! \brief The 1MB workspace size for integer attention auxiliary data. */
constexpr const int kIntAttnWorkspaceByte = 8 * 1024 * 1024;
/*! \brief The 128MB workspace size for floating-point attention auxiliary data. */
constexpr const int kFloatAttnWorkspaceByte = 768 * 1024 * 1024;
/*! \brief The id of the temporary logical page, which is useful for sliding window. */
constexpr const int kPagedKVCacheTempPageId = -1;

/*!
 * \brief The supported attention kinds in PagedKVCache.
 * "MHA" means multi-head attention, multi-query attention and grouped query attention in general.
 * "MLA" means multi-head latent attention.
 * "LinearAttn" means linear attention.
 */
enum class AttnKind : int {
  kMHA = 0,
  kMLA = 1,
  kLinearAttn = 2,
};

/*! \brief Given the attention kind and other metadata, return the one-layer KV cache shape. */
inline ffi::Shape GetKVCacheShape(AttnKind attn_kind, int64_t num_total_pages, int num_sequence,
                                  int64_t num_kv_heads, int64_t page_size, int64_t qk_head_dim,
                                  int64_t v_head_dim) {
  if (attn_kind == AttnKind::kMHA) {
    // Ignore v_head_dim since multi-head attention requires K/V to have the same head dim.
    return {num_total_pages, 2, num_kv_heads, page_size, qk_head_dim};
  } else if (attn_kind == AttnKind::kMLA) {
    return {num_total_pages, page_size, qk_head_dim};
  } else if (attn_kind == AttnKind::kLinearAttn) {
    return {num_sequence, num_kv_heads, qk_head_dim, v_head_dim};
  }
  ICHECK(false);
  return ffi::Shape();
}

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
   * It means the **first** sink size elements will be pinned
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
    start_pos = 0;
    sink_length = 0;
    sliding_window_offset = 0;
    parent_idx = -1;
    external_ref_cnt = 0;
  }
};

struct KVTransferMetadata {
  int64_t start = std::numeric_limits<int64_t>::max();
  std::vector<int64_t> remote_position_map;
  int32_t recver_pe_offset = -1;
  std::vector<int64_t> local_position_map;
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

  /*! \brief Whether the current appended tokens form a chain (not a tree). */
  bool is_chain = true;
  /*! \brief The token tree parent pointer array of the current appended tokens. */
  std::vector<int32_t> token_tree_parent_ptr;
  /*! \brief The depth of each node in the token tree. */
  std::vector<int32_t> token_tree_node_depths;
  /*! \brief The metadata of kv transfer*/
  KVTransferMetadata kv_transfer_metadata;
  /*!
   * \brief A boolean denoting whether the accepted token tree indices of
   * this sequence are committed
   */
  bool accepted_indices_committed = true;

  explicit Sequence(std::vector<Block>* global_block_pool, int32_t last_block_idx) {
    ++global_block_pool->at(last_block_idx).external_ref_cnt;
    this->last_block_idx = last_block_idx;
    int32_t block_ptr = last_block_idx;
    // Go through each block in the sequence, sum up the length.
    while (true) {
      const Block& block = global_block_pool->at(block_ptr);
      this->seq_length += block.seq_length;
      if (block.parent_idx == -1) {
        break;
      }
      block_ptr = block.parent_idx;
    }
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
 * \brief For the given list of sequences, check the block trace of
 * each sequence, and return the blocks ids used by the sequences
 * on each depth. And if the depth is larger than the kPagedKVCacheMaxBlockDepth,
 * the exceeding blocks will concatenate and output separately.
 * More precisely, the inner returned vector contains the block ids
 * used by the sequences on a certain depth (or "-1" if a sequence
 * has fewer depth). The outer returned vector contains the inner
 * vectors from the lowest depth to the highest depth.
 */
inline std::pair<std::vector<std::vector<int32_t>>, std::vector<std::vector<int32_t>>>
GetBlockIdsOnDepth(const std::vector<Sequence*>& sequences,
                   const std::vector<Block>& global_block_pool, int64_t batch_size) {
  // - Get the trace of each sequence.
  int64_t num_depths = 0;
  std::vector<std::vector<int32_t>> seq_block_traces;
  std::vector<std::vector<int32_t>> trailing_block_traces;
  seq_block_traces.reserve(batch_size);
  trailing_block_traces.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    std::vector<int32_t> trace = sequences[i]->GetBlockTrace(global_block_pool);
    if (static_cast<int>(trace.size()) <= kPagedKVCacheMaxBlockDepth) {
      seq_block_traces.push_back(std::vector<int32_t>(trace.begin(), trace.end()));
      trailing_block_traces.push_back({});
      num_depths = std::max(num_depths, static_cast<int64_t>(trace.size()));
    } else {
      seq_block_traces.push_back(
          std::vector<int32_t>(trace.begin(), trace.begin() + kPagedKVCacheMaxBlockDepth));
      trailing_block_traces.push_back(
          std::vector<int32_t>(trace.begin() + kPagedKVCacheMaxBlockDepth, trace.end()));
      num_depths = std::max(num_depths, static_cast<int64_t>(kPagedKVCacheMaxBlockDepth));
    }
  }

  // "Transpose" the traces, yielding the block ids used on each depth.
  std::vector<std::vector<int32_t>> block_ids_on_depths;
  block_ids_on_depths.reserve(num_depths);
  for (int d = 0; d < num_depths; ++d) {
    std::vector<int32_t> block_ids;
    block_ids.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      block_ids.push_back(d < static_cast<int>(seq_block_traces[i].size()) ? seq_block_traces[i][d]
                                                                           : -1);
    }
    block_ids_on_depths.push_back(std::move(block_ids));
  }
  return {block_ids_on_depths, trailing_block_traces};
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
inline std::pair<std::vector<std::pair<int32_t, int32_t>>, bool> GetChunkedBlockIds(
    const std::vector<int32_t>& block_ids, bool enable_coalesce, const IntTuple& append_lengths,
    const std::vector<Block>& global_block_pool, bool is_decode_request) {
  std::vector<std::pair<int32_t, int32_t>> uncoalesced_block_ids;
  std::vector<std::pair<int32_t, int32_t>> coalesced_block_ids;

  // Gather the number of pages before/after coalescing respectively.
  int cur_block_id = block_ids[0];
  int chunk_append_length = append_lengths[0];
  int page_counter_coalesced = 0;
  int page_counter_uncoalesced =
      block_ids[0] != -1 ? global_block_pool[block_ids[0]].page_ids.size() : 0;
  for (int i = 1; i < static_cast<int>(block_ids.size()); ++i) {
    if (block_ids[i] != -1) {
      page_counter_uncoalesced += global_block_pool[block_ids[i]].page_ids.size();
    }
    uncoalesced_block_ids.emplace_back(block_ids[i - 1], append_lengths[i - 1]);
    if (block_ids[i] == cur_block_id) {
      chunk_append_length += append_lengths[i];
    } else {
      coalesced_block_ids.emplace_back(cur_block_id, chunk_append_length);
      if (cur_block_id != -1) {
        page_counter_coalesced += global_block_pool[cur_block_id].page_ids.size();
      }
      cur_block_id = block_ids[i];
      chunk_append_length = append_lengths[i];
    }
  }
  uncoalesced_block_ids.emplace_back(block_ids.back(), append_lengths.back());
  coalesced_block_ids.emplace_back(cur_block_id, chunk_append_length);
  if (cur_block_id != -1) {
    page_counter_coalesced += global_block_pool[cur_block_id].page_ids.size();
  }
  double coalesce_ratio =
      page_counter_coalesced > 0 ? 1.0 * page_counter_uncoalesced / page_counter_coalesced : 0.0;
  // Do not coalesce and use batch decode kernel when coalesce ratio is small.
  bool use_decode_kernel = is_decode_request && coalesce_ratio < 32;
  return {use_decode_kernel || !enable_coalesce ? uncoalesced_block_ids : coalesced_block_ids,
          use_decode_kernel};
}

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
 * \brief The class of host memory int32 vector in "std::vector" interface.
 * This vector allocates static memory on the specified host memory
 * at the time of construction.
 */
class HostMemoryVector {
 public:
  HostMemoryVector() = default;
  HostMemoryVector(const HostMemoryVector&) = delete;
  HostMemoryVector(HostMemoryVector&& other) = default;
  HostMemoryVector& operator=(const HostMemoryVector&) = delete;
  HostMemoryVector& operator=(HostMemoryVector&& other) = default;

  explicit HostMemoryVector(int64_t reserved_size, DLDataType dtype, Device device)
      : reserved_size_(reserved_size) {
    ICHECK(DataType(dtype) == DataType::Int(32));
    data_ = NDArray::Empty({reserved_size}, dtype, device);
  }

  void push_back(int32_t value) {
    ICHECK_LE(current_size_, reserved_size_);
    if (current_size_ == reserved_size_) {
      reserved_size_ *= 2;
      NDArray new_data = NDArray::Empty({reserved_size_}, data_->dtype, data_->device);
      std::memcpy(new_data->data, data_->data, current_size_ * DataType(data_->dtype).bytes());
      data_ = new_data;
    }
    static_cast<int32_t*>(data_->data)[current_size_++] = value;
  }

  const int32_t& operator[](int64_t idx) const {
    ICHECK_GE(idx, 0) << "Index " << idx << " is negative.";
    ICHECK_LT(idx, current_size_) << "Index " << idx << " out of bounds " << current_size_;
    return static_cast<int32_t*>(data_->data)[idx];
  }

  int32_t back() const {
    ICHECK_GT(current_size_, 0) << "Vector is empty";
    return static_cast<int32_t*>(data_->data)[current_size_ - 1];
  }

  size_t size() const { return static_cast<size_t>(current_size_); }

  int32_t* data() const { return static_cast<int32_t*>(data_->data); }

  void clear() { current_size_ = 0; }

  /*! \brief Return the vector as an NDArray. */
  NDArray as_ndarray() { return data_.CreateView({current_size_}, data_->dtype); }

  IntTuple as_int_tuple() const {
    std::vector<int64_t> values;
    values.reserve(current_size_);
    for (int i = 0; i < current_size_; ++i) {
      values.push_back(static_cast<int32_t*>(data_->data)[i]);
    }
    return IntTuple(values);
  }

 private:
  int64_t reserved_size_ = 0;
  int64_t current_size_ = 0;
  NDArray data_{nullptr};
};

/*!
 * \brief The paged attention auxiliary data manager class.
 * This class manages all the int32 auxiliary data on GPU device, such as
 * page table, position arrays, etc..
 *
 * The core functions of this class is `CopyXXXAsync` and `CommitAttnAuxDataCopy`.
 * `CopyXXXAsync` takes the input data on CPU host, and copy the input data
 * to GPU in an asynchronous way, and returns the NDArray view of the data
 * on GPU device.
 *
 * Being asynchronous here means the `CopyXXXAsync` function may not perform
 * data copy from CPU to GPU at the time of being called. Therefore, the
 * returned NDArray view may have wrong result, until `CommitAttnAuxDataCopy` is
 * explicitly invoked and the data copy stream is synchronized.
 *
 * We design this manager class in order to reduce the data copy overhead.
 */
class PagedKVCacheAuxDataManager {
 public:
  PagedKVCacheAuxDataManager(DLDataType dtype_aux, Device device, Device preferred_host_device,
                             TVMStreamHandle copy_stream)
      : dtype_aux_(dtype_aux),
        device_(device),
        preferred_host_device_(preferred_host_device),
        copy_stream_(copy_stream) {
    ICHECK(DataType(dtype_aux) == DataType::Int(32));
  }

  virtual ~PagedKVCacheAuxDataManager() = default;
  /*! \brief Reset the attention auxiliary data status of copy manager. */
  virtual void ResetAttnAuxDataCopy() = 0;
  /*! \brief Copy the indptr array of append lengths after coalescing. (see GetChunkedBlockIds) */
  virtual NDArray CopyQOIndptrOnDepthAsync(HostMemoryVector* data, int depth) = 0;
  /*! \brief Copy the indptr array of page table. */
  virtual NDArray CopyPageIndptrOnDepthAsync(HostMemoryVector* data, int depth) = 0;
  /*! \brief Copy the indices array of page table. */
  virtual NDArray CopyPageIndicesOnDepthAsync(HostMemoryVector* data, int depth) = 0;
  /*! \brief Copy the array of KV slot number used in the last page of the seq. */
  virtual NDArray CopyLastPageLenOnDepthAsync(HostMemoryVector* data, int depth) = 0;
  /*!
   * \brief Copy the length information of the sequences.
   * Each NDArray is in shape `(3, n)`. "n" is the number of sequences.
   * For a sequence "i", location
   * - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
   * - "(1, i)" is the starting offset of the sliding window in the seq,
   * - "(2, i)" is the attn sink length of the sequence.
   * \note When sliding window is not enabled, only the
   * "last_page_len" (a.k.a., the first "n" elements) will be effectively used.
   */
  virtual NDArray CopyLengthInfoOnDepthAsync(HostMemoryVector* last_page_len,
                                             HostMemoryVector* sliding_window_offset,
                                             HostMemoryVector* sink_size, int depth) = 0;
  /*! \brief Copy the k position offset of applying RoPE for each sequence. */
  virtual NDArray CopyKRoPEPosOffsetOnDepthAsync(HostMemoryVector* data, int depth) = 0;
  /*!
   * \brief Copy the append length indptr array on device.
   * \note Since the Q/K/V data may have raggedness in terms of lengths,
   * we represent the append lengths in CSR format.
   */
  virtual NDArray CopyCurAppendLengthIndptrAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the k position offset of applying RoPE for each sequence. */
  virtual NDArray CopyKRaggedRoPEPosOffsetAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the q position mapping of applying RoPE for each sequence. */
  virtual NDArray CopyQRoPEPosMapAsync(HostMemoryVector* data) = 0;
  /*!
   * \brief Copy the corresponding position in global KV cache (pages)
   * for each position along the length dimension of K/V data when
   * appending new K/V data.
   */
  virtual NDArray CopyAppendPositionMapAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the remote position map for KV transfer. */
  virtual NDArray CopyKVTransferRemotePositionMapAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the receiver id for KV transfer. */
  virtual NDArray CopyKVTransferRecverIDAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the local position map for KV page-to-page transfer. */
  virtual NDArray CopyKVTransferPage2PageLocalPositionMapAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the remote position map for KV page-to-page transfer. */
  virtual NDArray CopyKVTransferPage2PageRemotePositionMapAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the receiver id for KV page-to-page transfer. */
  virtual NDArray CopyKVTransferPage2PageRecverIDAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the tree attention mask. */
  virtual NDArray CopyTreeAttnMaskOnDepthAsync(HostMemoryVector* data, int depth) = 0;
  /*! \brief Copy the mn indptr of the tree attention mask. */
  virtual NDArray CopyTreeAttnMNIndptrOnDepthAsync(HostMemoryVector* data, int depth) = 0;
  /*! \brief Commit all the attention auxiliary data copy operations since the last commit. */
  virtual void CommitAttnAuxDataCopy() = 0;

  /*! \brief Reset the compact KV auxiliary data status of copy manager. */
  virtual void ResetCompactKVAuxDataCopy() = 0;
  /*! \brief Copy the length indptr array of KV data copy for each sequence. */
  virtual NDArray CopyCommitLengthIndptrAsync(HostMemoryVector* data) = 0;
  /*! \brief Copy the src/dst position arrays for each sequence. */
  virtual NDArray CopyCommitSrcDstPosInPageTableAsync(HostMemoryVector* src_data,
                                                      HostMemoryVector* dst_data) = 0;
  /*! \brief Commit all the compact KV auxiliary data copy operations since the last commit. */
  virtual void CommitCompactKVAuxDataCopy() = 0;

 protected:
  /*! \brief The dtype of the auxiliary data. It is expected to be int32. */
  const DLDataType dtype_aux_;
  /*! \brief The device this PagedKVCache runs on. */
  const Device device_;
  /*! \brief The preferred host device. */
  const Device preferred_host_device_;
  /*! \brief The device stream for copying auxiliary data structure to GPU. */
  const TVMStreamHandle copy_stream_;
};

/*!
 * \brief The plain auxiliary data manager class.
 * It simply issues one host-to-device copy operation for each `CopyXXXAsync`.
 */
class PlainPagedKVCacheAuxDataManager : public PagedKVCacheAuxDataManager {
 public:
  explicit PlainPagedKVCacheAuxDataManager(int64_t reserved_num_seqs, int64_t num_total_pages,
                                           int64_t prefill_chunk_size, DLDataType dtype_aux,
                                           Device device, Device preferred_host_device,
                                           TVMStreamHandle copy_stream)
      : PagedKVCacheAuxDataManager(dtype_aux, device, preferred_host_device, copy_stream) {
    for (int d = 0; d < kPagedKVCacheMaxBlockDepth; ++d) {
      qo_indptr_on_depths_device_.push_back(
          NDArray::Empty({reserved_num_seqs + 1}, dtype_aux_, device));
      page_indptr_on_depths_device_.push_back(
          NDArray::Empty({reserved_num_seqs + 1}, dtype_aux_, device));
      page_indices_on_depths_device_.push_back(
          NDArray::Empty({num_total_pages}, dtype_aux_, device));
      length_info_on_depths_device_.push_back(
          NDArray::Empty({3, reserved_num_seqs}, dtype_aux_, device));
      k_rope_pos_offset_on_depths_device_.push_back(
          NDArray::Empty({reserved_num_seqs}, dtype_aux_, device));
      tree_attn_mask_device_.push_back(NDArray::Empty(
          {kTreeAttnMaxTreeSize * kTreeAttnMaxTreeSize * reserved_num_seqs}, dtype_aux_, device));
      tree_attn_mn_indptr_device_.push_back(
          NDArray::Empty({reserved_num_seqs + 1}, dtype_aux_, device));
    }
    cur_append_length_indptr_device_ = NDArray::Empty({reserved_num_seqs + 1}, dtype_aux_, device);
    k_ragged_rope_pos_offset_device_ = NDArray::Empty({reserved_num_seqs}, dtype_aux_, device);
    q_rope_position_map_device_ = NDArray::Empty({prefill_chunk_size}, dtype_aux_, device);
    append_position_map_device_ = NDArray::Empty({prefill_chunk_size}, dtype_aux_, device);
    kv_transfer_remote_position_map_device =
        NDArray::Empty({prefill_chunk_size}, dtype_aux_, device);
    kv_transfer_recver_id_device = NDArray::Empty({prefill_chunk_size}, dtype_aux_, device);
    kv_transfer_page_to_page_local_position_map_device =
        kv_transfer_page_to_page_remote_position_map_device =
            NDArray::Empty({prefill_chunk_size}, dtype_aux_, device);
    kv_transfer_page_to_page_recver_id_device =
        NDArray::Empty({prefill_chunk_size}, dtype_aux_, device);
    commit_copy_length_indptr_device_ = NDArray::Empty({reserved_num_seqs + 1}, dtype_aux_, device);
    commit_copy_src_dst_pos_in_page_table_device_ =
        NDArray::Empty({2, std::min(kTreeAttnMaxTreeSize * reserved_num_seqs, prefill_chunk_size)},
                       dtype_aux_, device);
  }

  // The reset of the plain auxiliary data manager is no-op.
  void ResetAttnAuxDataCopy() final {}
  NDArray CopyQOIndptrOnDepthAsync(HostMemoryVector* data, int depth) final {
    NDArray view = qo_indptr_on_depths_device_[depth].CreateView(
        {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyPageIndptrOnDepthAsync(HostMemoryVector* data, int depth) final {
    NDArray view = page_indptr_on_depths_device_[depth].CreateView(
        {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyPageIndicesOnDepthAsync(HostMemoryVector* data, int depth) final {
    NDArray view = page_indices_on_depths_device_[depth].CreateView(
        {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyLastPageLenOnDepthAsync(HostMemoryVector* data, int depth) final {
    NDArray view = length_info_on_depths_device_[depth].CreateView(
        {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyKRoPEPosOffsetOnDepthAsync(HostMemoryVector* data, int depth) final {
    NDArray view = k_rope_pos_offset_on_depths_device_[depth].CreateView(
        {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyCurAppendLengthIndptrAsync(HostMemoryVector* data) final {
    NDArray view = cur_append_length_indptr_device_.CreateView({static_cast<int64_t>(data->size())},
                                                               dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyKRaggedRoPEPosOffsetAsync(HostMemoryVector* data) final {
    NDArray view = k_ragged_rope_pos_offset_device_.CreateView({static_cast<int64_t>(data->size())},
                                                               dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyQRoPEPosMapAsync(HostMemoryVector* data) final {
    NDArray view =
        q_rope_position_map_device_.CreateView({static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyAppendPositionMapAsync(HostMemoryVector* data) final {
    NDArray view =
        append_position_map_device_.CreateView({static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyKVTransferRemotePositionMapAsync(HostMemoryVector* data) final {
    NDArray view = kv_transfer_remote_position_map_device.CreateView(
        {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyKVTransferRecverIDAsync(HostMemoryVector* data) final {
    NDArray view =
        kv_transfer_recver_id_device.CreateView({static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyKVTransferPage2PageLocalPositionMapAsync(HostMemoryVector* data) final {
    NDArray view = kv_transfer_page_to_page_local_position_map_device.CreateView(
        {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyKVTransferPage2PageRemotePositionMapAsync(HostMemoryVector* data) final {
    NDArray view = kv_transfer_page_to_page_remote_position_map_device.CreateView(
        {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyKVTransferPage2PageRecverIDAsync(HostMemoryVector* data) final {
    NDArray view = kv_transfer_page_to_page_recver_id_device.CreateView(
        {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }

  NDArray CopyTreeAttnMaskOnDepthAsync(HostMemoryVector* data, int depth) final {
    NDArray view =
        tree_attn_mask_device_[depth].CreateView({static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyTreeAttnMNIndptrOnDepthAsync(HostMemoryVector* data, int depth) final {
    NDArray view = tree_attn_mn_indptr_device_[depth].CreateView(
        {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }

  NDArray CopyLengthInfoOnDepthAsync(HostMemoryVector* last_page_len,
                                     HostMemoryVector* sliding_window_offset,
                                     HostMemoryVector* sink_size, int depth) final {
    int n_elem = last_page_len->size();
    ICHECK_GT(n_elem, 0);
    NDArray view = length_info_on_depths_device_[depth].CreateView({3, n_elem}, dtype_aux_);
    ffi::Shape copy_shape{n_elem};
    CopyVecDataToArray(view, last_page_len->data(), copy_shape);
    CopyVecDataToArray(view, sliding_window_offset->data(), copy_shape,
                       /*dst_elem_offset=*/n_elem);
    CopyVecDataToArray(view, sink_size->data(), copy_shape,
                       /*dst_elem_offset=*/2 * n_elem);
    return view;
  }

  // The commit of the plain auxiliary data manager is no-op.
  void CommitAttnAuxDataCopy() final {}

  // The reset of the plain auxiliary data manager is no-op.
  void ResetCompactKVAuxDataCopy() final {}

  NDArray CopyCommitLengthIndptrAsync(HostMemoryVector* data) final {
    NDArray view = commit_copy_length_indptr_device_.CreateView(
        {static_cast<int64_t>(data->size())}, dtype_aux_);
    CopyVecDataToArray(view, data->data());
    return view;
  }
  NDArray CopyCommitSrcDstPosInPageTableAsync(HostMemoryVector* src_data,
                                              HostMemoryVector* dst_data) final {
    int n_elem = src_data->size();
    ICHECK_GT(n_elem, 0);
    NDArray view =
        commit_copy_src_dst_pos_in_page_table_device_.CreateView({2, n_elem}, dtype_aux_);
    ffi::Shape copy_shape{n_elem};
    CopyVecDataToArray(view, src_data->data(), copy_shape);
    CopyVecDataToArray(view, dst_data->data(), copy_shape,
                       /*dst_elem_offset=*/n_elem);
    return view;
  }

  // The commit of the plain auxiliary data manager is no-op.
  void CommitCompactKVAuxDataCopy() final {}

 private:
  /*!
   * \brief Copy a vector of data to the input NDArray.
   * It optionally supports specifying the shape of copy and the element
   * offset to the destination NDArray.
   */
  void CopyVecDataToArray(NDArray array, int32_t* vec_data,
                          Optional<ffi::Shape> shape = std::nullopt, int dst_elem_offset = 0) {
    if (array->shape[0] == 0) {
      return;
    }
    DLTensor copy_dst = *array.operator->();
#if defined(OPENCL_ENABLE_HOST_PTR)
    tvm::runtime::cl::OpenCLWorkspace* workspace = tvm::runtime::cl::OpenCLWorkspace::Global();
    if (workspace->IsOpenCLDevice(copy_dst.device)) {
      void* nptr = workspace->GetNativePtr(array);
      uint64_t copy_size;
      if (shape.defined()) {
        ICHECK_EQ(shape.value().size(), 1);
        copy_size = shape.value()->data[0] * sizeof(int32_t);
      } else {
        copy_size = DeviceAPI::Get(array->device)->GetDataSize(*array.operator->());
      }
      memcpy(static_cast<char*>(nptr) + dst_elem_offset * sizeof(int32_t), vec_data, copy_size);
      return;
    }
#endif

    if (shape.defined()) {
      ICHECK_EQ(shape.value().size(), 1);
      copy_dst.ndim = 1;
      copy_dst.shape = const_cast<int64_t*>(shape.value()->data);
    }
    copy_dst.byte_offset = dst_elem_offset * sizeof(int32_t);

    DLTensor copy_src;
    copy_src.data = vec_data;
    copy_src.device = preferred_host_device_;
    copy_src.ndim = 1;
    copy_src.dtype = array->dtype;
    copy_src.shape = copy_dst.shape;
    copy_src.strides = nullptr;
    copy_src.byte_offset = 0;
    NDArray::CopyFromTo(&copy_src, &copy_dst, copy_stream_);
  }

  std::vector<NDArray> qo_indptr_on_depths_device_;
  std::vector<NDArray> page_indptr_on_depths_device_;
  std::vector<NDArray> page_indices_on_depths_device_;
  std::vector<NDArray> length_info_on_depths_device_;
  std::vector<NDArray> k_rope_pos_offset_on_depths_device_;
  std::vector<NDArray> tree_attn_mask_device_;
  std::vector<NDArray> tree_attn_mn_indptr_device_;
  NDArray cur_append_length_indptr_device_;
  NDArray k_ragged_rope_pos_offset_device_;
  NDArray q_rope_position_map_device_;
  NDArray append_position_map_device_;
  NDArray kv_transfer_remote_position_map_device;
  NDArray kv_transfer_recver_id_device;
  NDArray kv_transfer_page_to_page_local_position_map_device;
  NDArray kv_transfer_page_to_page_remote_position_map_device;
  NDArray kv_transfer_page_to_page_recver_id_device;
  NDArray commit_copy_length_indptr_device_;
  NDArray commit_copy_src_dst_pos_in_page_table_device_;
};

/*!
 * \brief The cached auxiliary data manager class.
 * It allocates a large on-device array to store all the auxiliary data.
 * For each `CopyXXXAsync`, it copies the input data to a local cache on host.
 * In `CommitAttnAuxDataCopy`, it copies all the data in the local cache to the device
 * array for a single time, and thus reduce the number of host-to-device copies needed.
 */
class CachedPagedKVCacheAuxDataManager : public PagedKVCacheAuxDataManager {
 public:
  explicit CachedPagedKVCacheAuxDataManager(int64_t reserved_num_seqs, int64_t num_total_pages,
                                            int64_t prefill_chunk_size, DLDataType dtype_aux,
                                            Device device, Device preferred_host_device,
                                            TVMStreamHandle copy_stream)
      : PagedKVCacheAuxDataManager(dtype_aux, device, preferred_host_device, copy_stream),
        elem_byte_size_((dtype_aux.bits * dtype_aux.lanes + 7) / 8),
        offset_alignment_(cuda_byte_alignment_ / elem_byte_size_) {
    // - Calculate cache size of all the attention auxiliary arrays in
    // local cache and the large on-device array.
    int64_t attn_aux_data_cache_size =
        CalculateAttnAuxDataCacheSize(reserved_num_seqs, num_total_pages, prefill_chunk_size);
    // - Initialize the host auxiliary data buffer.
    merged_attn_aux_data_host_ =
        HostMemoryVector(attn_aux_data_cache_size, dtype_aux, preferred_host_device);
    // - Initialize the device auxiliary data buffer.
    merged_attn_aux_data_device_ = NDArray::Empty({attn_aux_data_cache_size}, dtype_aux, device);

    // - Calculate cache size of all the compact KV auxiliary arrays in
    // local cache and the large on-device array.
    int64_t compact_kv_aux_data_cache_size =
        CalculateCompactKVAuxDataCacheSize(reserved_num_seqs, prefill_chunk_size);
    // - Initialize the host auxiliary data buffer.
    merged_compact_kv_aux_data_host_ =
        HostMemoryVector(compact_kv_aux_data_cache_size, dtype_aux, preferred_host_device);
    merged_compact_kv_aux_data_device_ =
        NDArray::Empty({compact_kv_aux_data_cache_size}, dtype_aux, device);
  }

  void ResetAttnAuxDataCopy() final { attn_aux_data_copy_offset_ = 0; }
  NDArray CopyQOIndptrOnDepthAsync(HostMemoryVector* data, int depth) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyPageIndptrOnDepthAsync(HostMemoryVector* data, int depth) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyPageIndicesOnDepthAsync(HostMemoryVector* data, int depth) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyLastPageLenOnDepthAsync(HostMemoryVector* data, int depth) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyKRoPEPosOffsetOnDepthAsync(HostMemoryVector* data, int depth) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyCurAppendLengthIndptrAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyKRaggedRoPEPosOffsetAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyQRoPEPosMapAsync(HostMemoryVector* data) final { return CopyAttnAuxVecToCache(data); }
  NDArray CopyAppendPositionMapAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyKVTransferRemotePositionMapAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyKVTransferRecverIDAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyKVTransferPage2PageLocalPositionMapAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyKVTransferPage2PageRemotePositionMapAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyKVTransferPage2PageRecverIDAsync(HostMemoryVector* data) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyTreeAttnMaskOnDepthAsync(HostMemoryVector* data, int depth) final {
    NDArray mask_1d = CopyAttnAuxVecToCache(data);
    return mask_1d.CreateView({static_cast<int64_t>(data->size() / 2), 2}, mask_1d->dtype);
  }
  NDArray CopyTreeAttnMNIndptrOnDepthAsync(HostMemoryVector* data, int depth) final {
    return CopyAttnAuxVecToCache(data);
  }
  NDArray CopyLengthInfoOnDepthAsync(HostMemoryVector* last_page_len,
                                     HostMemoryVector* sliding_window_offset,
                                     HostMemoryVector* sink_size, int depth) final {
    int64_t n_elem = last_page_len->size();
    std::memcpy(merged_attn_aux_data_host_.data() + attn_aux_data_copy_offset_,
                last_page_len->data(), n_elem * elem_byte_size_);
    std::memcpy(merged_attn_aux_data_host_.data() + attn_aux_data_copy_offset_ + n_elem,
                sliding_window_offset->data(), n_elem * elem_byte_size_);
    std::memcpy(merged_attn_aux_data_host_.data() + attn_aux_data_copy_offset_ + 2 * n_elem,
                sink_size->data(), n_elem * elem_byte_size_);
    NDArray view = merged_attn_aux_data_device_.CreateView(
        {3, n_elem}, dtype_aux_, attn_aux_data_copy_offset_ * elem_byte_size_);
    attn_aux_data_copy_offset_ += CeilDivElemAlignment(3 * n_elem);
    return view;
  }

  void CommitAttnAuxDataCopy() final {
    std::vector<int64_t> copy_shape{attn_aux_data_copy_offset_};
    DLTensor copy_dst;
    copy_dst.data = merged_attn_aux_data_device_->data;
    copy_dst.device = device_;
    copy_dst.ndim = 1;
    copy_dst.dtype = dtype_aux_;
    copy_dst.shape = copy_shape.data();
    copy_dst.strides = nullptr;
    copy_dst.byte_offset = 0;

    DLTensor copy_src = copy_dst;
    copy_src.data = merged_attn_aux_data_host_.data();
    copy_src.device = Device{kDLCPU, 0};
    NDArray::CopyFromTo(&copy_src, &copy_dst, copy_stream_);
  }

  void ResetCompactKVAuxDataCopy() final { compact_kv_aux_data_copy_offset_ = 0; }

  NDArray CopyCommitLengthIndptrAsync(HostMemoryVector* data) final {
    return CopyCompactKVAuxVecToCache(data);
  }
  NDArray CopyCommitSrcDstPosInPageTableAsync(HostMemoryVector* src_data,
                                              HostMemoryVector* dst_data) final {
    int64_t n_elem = src_data->size();
    std::memcpy(merged_compact_kv_aux_data_host_.data() + compact_kv_aux_data_copy_offset_,
                src_data->data(), n_elem * elem_byte_size_);
    std::memcpy(merged_compact_kv_aux_data_host_.data() + compact_kv_aux_data_copy_offset_ + n_elem,
                dst_data->data(), n_elem * elem_byte_size_);
    NDArray view = merged_compact_kv_aux_data_device_.CreateView(
        {2, n_elem}, dtype_aux_, compact_kv_aux_data_copy_offset_ * elem_byte_size_);
    compact_kv_aux_data_copy_offset_ += CeilDivElemAlignment(2 * n_elem);
    return view;
  }

  void CommitCompactKVAuxDataCopy() final {
    std::vector<int64_t> copy_shape{compact_kv_aux_data_copy_offset_};
    DLTensor copy_dst;
    copy_dst.data = merged_compact_kv_aux_data_device_->data;
    copy_dst.device = device_;
    copy_dst.ndim = 1;
    copy_dst.dtype = dtype_aux_;
    copy_dst.shape = copy_shape.data();
    copy_dst.strides = nullptr;
    copy_dst.byte_offset = 0;

    DLTensor copy_src = copy_dst;
    copy_src.data = merged_compact_kv_aux_data_host_.data();
    copy_src.device = Device{kDLCPU, 0};
    NDArray::CopyFromTo(&copy_src, &copy_dst, copy_stream_);
  }

 private:
  /*!
   * \brief Calculate the start element offsets of the auxiliary arrays in the local cache.
   * \return Return the local cache size (total number of elements in the local cache).
   */
  int64_t CalculateAttnAuxDataCacheSize(int64_t reserved_num_seqs, int64_t num_total_pages,
                                        int64_t prefill_chunk_size) {
    int64_t cache_size = 0;
    // - Array size of the arrays that every depth has.
    // Corresponding to the following arrays respectively
    //  - qo_indptr_in_depth
    //  - page_indptr_in_depth
    //  - page_indices_in_depth
    //  - length_info_in_depth
    //  - k_rope_pos_offset_in_depth
    cache_size += CeilDivElemAlignment(reserved_num_seqs + 1);
    cache_size += CeilDivElemAlignment(reserved_num_seqs + 1);
    cache_size += CeilDivElemAlignment(num_total_pages);
    cache_size += CeilDivElemAlignment(3 * reserved_num_seqs);
    cache_size += CeilDivElemAlignment(reserved_num_seqs);
    cache_size *= kPagedKVCacheMaxBlockDepth;

    // - Array size of other arrays.
    // Corresponding to the following arrays respectively
    //  - cur_append_length_indptr
    //  - k_ragged_rope_pos_offset
    //  - q_rope_position_map
    //  - append_position_map
    //  - kv_transfer_remote_position_map
    //  - kv_transfer_recver_id
    //  - kv_transfer_page_to_page_local_position_map
    //  - kv_transfer_page_to_page_remote_position_map
    //  - kv_transfer_page_to_page_recver_id
    //  - tree_attn_mask
    //  - tree_attn_mn_indptr
    cache_size += CeilDivElemAlignment(reserved_num_seqs + 1);
    cache_size += CeilDivElemAlignment(reserved_num_seqs);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size += CeilDivElemAlignment(prefill_chunk_size);
    cache_size +=
        CeilDivElemAlignment(kTreeAttnMaxTreeSize * kTreeAttnMaxTreeSize * reserved_num_seqs);
    cache_size += CeilDivElemAlignment(reserved_num_seqs + 1);

    return cache_size;
  }

  int64_t CalculateCompactKVAuxDataCacheSize(int64_t reserved_num_seqs,
                                             int64_t prefill_chunk_size) {
    int64_t cache_size = 0;
    // Corresponding to the following arrays respectively
    //  - commit_copy_length_indptr
    //  - commit_copy_src_dst_pos_in_page_table
    cache_size += CeilDivElemAlignment(reserved_num_seqs + 1);
    cache_size += CeilDivElemAlignment(
        2 * std::min(kTreeAttnMaxTreeSize * reserved_num_seqs, prefill_chunk_size));

    return cache_size;
  }

  /*!
   * \brief Copy the input data to the cache at the given offset.
   * And return the NDArray view of the cache starting at the offset.
   */
  NDArray CopyAttnAuxVecToCache(HostMemoryVector* data) {
    int64_t n_elem = data->size();
    std::memcpy(merged_attn_aux_data_host_.data() + attn_aux_data_copy_offset_, data->data(),
                n_elem * elem_byte_size_);
    NDArray view = merged_attn_aux_data_device_.CreateView(
        {n_elem}, dtype_aux_, attn_aux_data_copy_offset_ * elem_byte_size_);
    attn_aux_data_copy_offset_ += CeilDivElemAlignment(n_elem);
    return view;
  }

  NDArray CopyCompactKVAuxVecToCache(HostMemoryVector* data) {
    int64_t n_elem = data->size();
    std::memcpy(merged_compact_kv_aux_data_host_.data() + compact_kv_aux_data_copy_offset_,
                data->data(), n_elem * elem_byte_size_);
    NDArray view = merged_compact_kv_aux_data_device_.CreateView(
        {n_elem}, dtype_aux_, compact_kv_aux_data_copy_offset_ * elem_byte_size_);
    compact_kv_aux_data_copy_offset_ += CeilDivElemAlignment(n_elem);
    return view;
  }

  /*! \brief For safety, we align the start offset of the arrays to `offset_alignment`. */
  int64_t CeilDivElemAlignment(int n) {
    return (n + offset_alignment_ - 1) / offset_alignment_ * offset_alignment_;
  }

  const int64_t cuda_byte_alignment_ = 16;
  const int64_t elem_byte_size_;
  const int64_t offset_alignment_;

  int64_t attn_aux_data_copy_offset_ = 0;
  int64_t compact_kv_aux_data_copy_offset_ = 0;
  HostMemoryVector merged_attn_aux_data_host_;
  HostMemoryVector merged_compact_kv_aux_data_host_;
  NDArray merged_attn_aux_data_device_;
  NDArray merged_compact_kv_aux_data_device_;
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_RELAX_VM_ATTN_UTILS_H_
