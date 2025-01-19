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
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kv_state.h"
#if defined(OPENCL_ENABLE_HOST_PTR)
#include "../opencl/opencl_common.h"
#endif

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
constexpr const int kPagedKVCacheMaxBlockDepth = 2;
/*! \brief The maximum tree size of a single sequence in tree attention. */
constexpr const int kTreeAttnMaxTreeSize = 256;
/*! \brief The 1MB workspace size for integer attention auxiliary data. */
constexpr const int kIntAttnWorkspaceByte = 1 * 1024 * 1024;
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

ShapeTuple GetKVCacheShape(AttnKind attn_kind, int64_t num_total_pages, int num_sequence,
                           int64_t num_kv_heads, int64_t page_size, int64_t qk_head_dim,
                           int64_t v_head_dim, int64_t qk_rope_head_dim) {
  if (attn_kind == AttnKind::kMHA) {
    // Ignore v_head_dim since multi-head attention requires K/V to have the same head dim.
    return {num_total_pages, 2, num_kv_heads, page_size, qk_head_dim};
  } else if (attn_kind == AttnKind::kMLA) {
    return {num_total_pages, page_size, qk_head_dim};
  } else if (attn_kind == AttnKind::kLinearAttn) {
    return {num_sequence, num_kv_heads, qk_head_dim, v_head_dim};
  }
  ICHECK(false);
  throw;
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
    ShapeTuple copy_shape{n_elem};
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
    ShapeTuple copy_shape{n_elem};
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
  void CopyVecDataToArray(NDArray array, int32_t* vec_data, Optional<ShapeTuple> shape = NullOpt,
                          int dst_elem_offset = 0) {
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
      copy_dst.shape = shape.value()->data;
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
  /*! \brief The beginning layer id offset. */
  const int64_t layer_id_begin_offset_;
  /*! \brief The number of query/output heads in the model. */
  const int64_t num_qo_heads_;
  /*! \brief The number of key/value heads in the model. */
  const int64_t num_kv_heads_;
  /*! \brief The number of features each head has. */
  const int64_t qk_head_dim_;
  /*!
   * \brief The number of features each head has for V.
   * For layers that use multi-head attention, this field is overriden by qk_head_dim.
   */
  const int64_t v_head_dim_;
  /*!
   * \brief The number of features each head has for RoPE in multi-head latent attention.
   * This field is ignored for non-MLA.
   */
  const int64_t qk_rope_head_dim_;
  /*! \brief The number of total pages allocated in KV cache. */
  const int64_t num_total_pages_;
  /*! \brief The maximum total sequence length in a prefill. */
  const int64_t prefill_chunk_size_;
  /*! \brief A boolean flag indicating if the KV cache supports sliding window. */
  const bool support_sliding_window_;
  /*! \brief The attention kinds for each layer. */
  const std::vector<AttnKind> attn_kinds_;

  /*! \brief The RoPE application mode of KV cache.*/
  const RoPEMode rope_mode_;
  /*! \brief The RoPE scale. */
  const double rotary_scale_;
  /*! \brief The RoPE theta. */
  const double rotary_theta_;
  /*! \brief The optional RoPE extension factors for RoPE scaling. */
  const Optional<NDArray> rope_ext_factors_;

  /*! \brief We fix int32 to be the index dtype of auxiliary data. */
  const DLDataType dtype_aux_ = DLDataType(DataType::Int(32, 1));

  /********************* Page Structures *********************/

  /*!
   * \brief The KV data managed by the KV cache.
   * If KV transfer function is specifed, pages_ will be allocated by NVSHMEM as a whole NDArray.
   * pages_ will contain tensor view of each layer.
   * Otherwise, pages_ has `num_layers` NDArrays, each of them
   * has layout (num_pages, 2, num_heads, page_size, qk_head_dim).
   * Along on the "2" dimension, index 0 stands for K and 1 stands for V.
   */
  std::vector<NDArray> pages_;
  /*! \brief The whole KV cache allocated by NVSHMEM*/
  NDArray nvshmem_pages_;
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
  /*! \brief The ids of the sequences in the current round of forwarding. */
  IntTuple cur_seq_ids_;
  /*! \brief The append lengths of the sequences in the current round of forwarding. */
  IntTuple cur_append_lengths_;
  /*! \brief Whether the current batch of sequences are token chains (not token trees). */
  std::vector<bool> is_chain_on_depths_;
  /*! \brief Number of fork depth in the current round of forward. */
  int num_depths_;
  /*! \brief Whether to compute attention after appending KV into cache or not. */
  bool append_before_attn_;
  /*! \brief Whether to use decode kernel for each depth. (see GetChunkedBlockIds) */
  std::vector<bool> use_decode_kernel_;
  /*! \brief Whether the attention request is a decode request, set in BeginForwardFunction. */
  bool is_decode_request_;
  /*! \brief The KV transfer recver disco group's PE offset in this forward.
             If no KV is transfered, recver is -1.
             Assume that all the KV are transfered to the same recver in the forward.
             todo: support multiple recver. */
  bool transfer_kv_;
  bool page_to_page_transfer_kv_;
  /*! \brief The auxiliary data manager for attention. */
  std::unique_ptr<PagedKVCacheAuxDataManager> aux_data_manager_;

  // Temporary arrays to store intermediate attention results.
  NDArray temp_attn_q_device_;
  NDArray temp_attn_k_device_;
  NDArray temp_attn_v_device_;
  NDArray temp_attn_output_device_;
  NDArray temp_attn_scores_device_;
  NDArray merged_attn_scores_device_;
  std::vector<NDArray> temp_int_attn_workspace_;
  NDArray temp_float_attn_workspace_;

  //-------------------------------------------
  // Below are the auxiliary data structure on CPU.
  // We make them class members to avoid repetitive allocation time in BeginForward.
  //-------------------------------------------
  std::vector<HostMemoryVector> qo_indptr_on_depths_host_;
  std::vector<HostMemoryVector> page_indptr_on_depths_host_;
  std::vector<HostMemoryVector> page_indices_on_depths_host_;
  std::vector<HostMemoryVector> last_page_len_on_depths_host_;
  std::vector<HostMemoryVector> sliding_window_offset_on_depths_host_;
  std::vector<HostMemoryVector> sink_size_on_depths_host_;
  std::vector<HostMemoryVector> k_rope_pos_offset_on_depths_host_;
  HostMemoryVector k_ragged_rope_pos_offset_host_;
  HostMemoryVector q_rope_position_map_host_;
  HostMemoryVector append_position_map_host_;
  HostMemoryVector cur_append_lengths_indptr_host_;
  std::vector<HostMemoryVector> tree_attn_mask_host_;
  std::vector<HostMemoryVector> tree_attn_mn_indptr_host_;
  HostMemoryVector commit_copy_length_indptr_host_;
  HostMemoryVector commit_copy_src_pos_in_page_table_host_;
  HostMemoryVector commit_copy_dst_pos_in_page_table_host_;
  HostMemoryVector kv_transfer_remote_position_map_host_;
  HostMemoryVector kv_transfer_recver_id_host_;
  HostMemoryVector kv_transfer_page_to_page_local_position_map_host_;
  HostMemoryVector kv_transfer_page_to_page_remote_position_map_host_;
  HostMemoryVector kv_transfer_page_to_page_recver_id_host_;

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
  NDArray kv_transfer_remote_position_map_view_;
  NDArray kv_transfer_recver_id_view_;
  NDArray kv_transfer_page_to_page_local_position_map_view_;
  NDArray kv_transfer_page_to_page_remote_position_map_view_;
  NDArray kv_transfer_page_to_page_recver_id_view_;
  NDArray temp_attn_output_view_;
  NDArray temp_attn_scores_view_;
  NDArray merged_attn_scores_view_;
  std::vector<NDArray> qo_indptr_on_depths_view_;
  std::vector<NDArray> page_indptr_on_depths_view_;
  std::vector<NDArray> page_indices_on_depths_view_;
  std::vector<NDArray> length_info_on_depths_view_;
  std::vector<NDArray> k_rope_pos_offset_view_;
  std::vector<NDArray> tree_attn_mask_view_;
  std::vector<NDArray> tree_attn_mn_indptr_view_;

  PackedFunc f_transpose_append_;
  PackedFunc f_transpose_append_mla_;
  Optional<PackedFunc> f_transfer_kv_;
  Optional<PackedFunc> f_transfer_kv_page_to_page_ = NullOpt;
  PackedFunc f_compact_copy_;
  PackedFunc f_attention_prefill_;
  PackedFunc f_attention_decode_;
  PackedFunc f_attention_prefill_sliding_window_;
  PackedFunc f_attention_decode_sliding_window_;
  PackedFunc f_attention_prefill_ragged_;
  PackedFunc f_attention_prefill_with_tree_mask_;
  PackedFunc f_attention_prefill_with_tree_mask_paged_kv_;
  Optional<PackedFunc> f_attention_prefill_ragged_begin_forward_;
  Optional<PackedFunc> f_attention_prefill_ragged_end_forward_;
  Optional<PackedFunc> f_attention_prefill_begin_forward_;
  Optional<PackedFunc> f_attention_prefill_end_forward_;
  Optional<PackedFunc> f_attention_decode_begin_forward_;
  Optional<PackedFunc> f_attention_decode_end_forward_;
  PackedFunc f_mla_prefill_;
  PackedFunc f_mla_decode_;
  PackedFunc f_mla_prefill_ragged_normal_;
  PackedFunc f_mla_prefill_ragged_absorbed_;
  PackedFunc f_merge_inplace_;
  PackedFunc f_split_rotary_;
  PackedFunc f_copy_single_page_;
  Optional<PackedFunc> f_debug_get_kv_;

  /*! \brief The device this PagedKVCache runs on. */
  Device device_;
  /*! \brief The device stream for the default computation operations. */
  TVMStreamHandle compute_stream_ = nullptr;
  /*! \brief The device stream for copying auxiliary data structure to GPU. */
  TVMStreamHandle copy_stream_ = nullptr;
  /*! \brief The device stream for KV transfer */
  TVMStreamHandle kv_transfer_stream_ = nullptr;

 public:
  /*! \brief Constructor. Take the cache configuration and initialize the NDArrays. */
  explicit PagedAttentionKVCacheObj(
      int64_t page_size, int64_t num_layers, int64_t layer_id_begin_offset,  //
      int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim, int64_t v_head_dim,
      int64_t qk_rope_head_dim, std::vector<AttnKind> attn_kinds, int64_t reserved_num_seqs,
      int64_t num_total_pages, int64_t prefill_chunk_size, bool support_sliding_window,
      RoPEMode rope_mode, double rotary_scale, double rotary_theta,
      Optional<NDArray> rope_ext_factors, bool enable_kv_transfer, DLDataType dtype, Device device,
      PackedFunc f_transpose_append, PackedFunc f_transpose_append_mla, PackedFunc f_compact_copy,
      PackedFunc f_attention_prefill, PackedFunc f_attention_decode,
      PackedFunc f_attention_prefill_sliding_window, PackedFunc f_attention_decode_sliding_window,
      PackedFunc f_attention_prefill_ragged, PackedFunc f_attention_prefill_with_tree_mask,
      PackedFunc f_attention_prefill_with_tree_mask_paged_kv,
      Optional<PackedFunc> f_attention_prefill_ragged_begin_forward,
      Optional<PackedFunc> f_attention_prefill_ragged_end_forward,
      Optional<PackedFunc> f_attention_prefill_begin_forward,
      Optional<PackedFunc> f_attention_prefill_end_forward,
      Optional<PackedFunc> f_attention_decode_begin_forward,
      Optional<PackedFunc> f_attention_decode_end_forward, PackedFunc f_mla_prefill,
      PackedFunc f_mla_decode, PackedFunc f_mla_prefill_ragged_normal,
      PackedFunc f_mla_prefill_ragged_absorbed, PackedFunc f_merge_inplace,
      PackedFunc f_split_rotary, PackedFunc f_copy_single_page, Optional<PackedFunc> f_debug_get_kv)
      : page_size_(page_size),
        num_layers_(num_layers),
        layer_id_begin_offset_(layer_id_begin_offset),
        num_qo_heads_(num_qo_heads),
        num_kv_heads_(num_kv_heads),
        qk_head_dim_(qk_head_dim),
        v_head_dim_(v_head_dim),
        qk_rope_head_dim_(qk_rope_head_dim),
        num_total_pages_(num_total_pages),
        prefill_chunk_size_(prefill_chunk_size),
        support_sliding_window_(support_sliding_window),
        attn_kinds_(std::move(attn_kinds)),
        rope_mode_(support_sliding_window && rope_mode != RoPEMode::kNone ? RoPEMode::kInline
                                                                          : rope_mode),
        rotary_scale_(rotary_scale),
        rotary_theta_(rotary_theta),
        rope_ext_factors_(std::move(rope_ext_factors)),
        f_transpose_append_(std::move(f_transpose_append)),
        f_transpose_append_mla_(std::move(f_transpose_append_mla)),
        f_compact_copy_(std::move(f_compact_copy)),
        f_attention_prefill_(std::move(f_attention_prefill)),
        f_attention_decode_(std::move(f_attention_decode)),
        f_attention_prefill_sliding_window_(std::move(f_attention_prefill_sliding_window)),
        f_attention_decode_sliding_window_(std::move(f_attention_decode_sliding_window)),
        f_attention_prefill_ragged_(std::move(f_attention_prefill_ragged)),
        f_attention_prefill_with_tree_mask_(std::move(f_attention_prefill_with_tree_mask)),
        f_attention_prefill_with_tree_mask_paged_kv_(
            std::move(f_attention_prefill_with_tree_mask_paged_kv)),
        f_attention_prefill_ragged_begin_forward_(
            std::move(f_attention_prefill_ragged_begin_forward)),
        f_attention_prefill_ragged_end_forward_(std::move(f_attention_prefill_ragged_end_forward)),
        f_attention_prefill_begin_forward_(std::move(f_attention_prefill_begin_forward)),
        f_attention_prefill_end_forward_(std::move(f_attention_prefill_end_forward)),
        f_attention_decode_begin_forward_(std::move(f_attention_decode_begin_forward)),
        f_attention_decode_end_forward_(std::move(f_attention_decode_end_forward)),
        f_mla_prefill_(std::move(f_mla_prefill)),
        f_mla_decode_(std::move(f_mla_decode)),
        f_mla_prefill_ragged_normal_(std::move(f_mla_prefill_ragged_normal)),
        f_mla_prefill_ragged_absorbed_(std::move(f_mla_prefill_ragged_absorbed)),
        f_merge_inplace_(std::move(f_merge_inplace)),
        f_split_rotary_(std::move(f_split_rotary)),
        f_copy_single_page_(std::move(f_copy_single_page)),
        f_debug_get_kv_(std::move(f_debug_get_kv)),
        device_(device) {
    // Note: For MLA, sliding window and disaggregation are disabled for now.
    if (std::find(attn_kinds_.begin(), attn_kinds_.end(), AttnKind::kMLA) != attn_kinds_.end()) {
      CHECK(!support_sliding_window_) << "Sliding window not supported yet for MLA";
      CHECK(!enable_kv_transfer) << "KV transfer not supported yet for MLA";
    }

    pages_.reserve(num_layers);
    if (enable_kv_transfer) {
      // For now, KV transfer only supports MHA.
      for (AttnKind attn_kind : attn_kinds_) {
        CHECK(attn_kind == AttnKind::kMHA);
      }
      CHECK(Registry::Get("runtime.disco.nvshmem.init_nvshmem") != nullptr)
          << "NVSHMEM is not enabled. Please make sure NVSHMEM is enabled when compiling TVM.";
      const PackedFunc* f_nvshmem_empty = runtime::Registry::Get("runtime.disco.nvshmem.empty");
      ICHECK_NOTNULL(f_nvshmem_empty);
      nvshmem_pages_ = (*f_nvshmem_empty)(
          ShapeTuple({num_layers, num_total_pages, 2, num_kv_heads, page_size, qk_head_dim}), dtype,
          device);
      for (int i = 0; i < num_layers; ++i) {
        pages_.push_back(nvshmem_pages_.CreateView(
            {num_total_pages_, 2, num_kv_heads_, page_size_, qk_head_dim_}, nvshmem_pages_->dtype,
            i * num_total_pages_ * 2 * num_kv_heads_ * page_size_ * qk_head_dim_ *
                nvshmem_pages_.DataType().bytes()));
      }

      const PackedFunc* f_transfer_kv_ptr = Registry::Get("nvshmem.KVTransfer");
      const PackedFunc* f_transfer_kv_page_to_page_ptr =
          Registry::Get("nvshmem.KVTransferPageToPage");
      ICHECK_NOTNULL(f_transfer_kv_ptr);
      ICHECK_NOTNULL(f_transfer_kv_page_to_page_ptr);
      f_transfer_kv_ = *f_transfer_kv_ptr;
      f_transfer_kv_page_to_page_ = *f_transfer_kv_page_to_page_ptr;
    } else {
      for (int i = 0; i < num_layers; ++i) {
        ShapeTuple kv_cache_shape = GetKVCacheShape(
            attn_kinds_[layer_id_begin_offset_ + i], num_total_pages, reserved_num_seqs,
            num_kv_heads, page_size, qk_head_dim, v_head_dim, qk_rope_head_dim);
        pages_.push_back(NDArray::Empty(kv_cache_shape, dtype, device));
      }
    }

    // Allocate the host memory.
    Device preferred_host_device = GetPreferredHostDevice(device);
    for (int d = 0; d < kPagedKVCacheMaxBlockDepth; ++d) {
      qo_indptr_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs + 1, dtype_aux_, preferred_host_device));
      page_indptr_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs + 1, dtype_aux_, preferred_host_device));
      page_indices_on_depths_host_.push_back(
          HostMemoryVector(num_total_pages, dtype_aux_, preferred_host_device));
      last_page_len_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs, dtype_aux_, preferred_host_device));
      sliding_window_offset_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs, dtype_aux_, preferred_host_device));
      sink_size_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs, dtype_aux_, preferred_host_device));
      k_rope_pos_offset_on_depths_host_.push_back(
          HostMemoryVector(reserved_num_seqs, dtype_aux_, preferred_host_device));
      tree_attn_mask_host_.push_back(HostMemoryVector(kTreeAttnMaxTreeSize * 2 * reserved_num_seqs,
                                                      dtype_aux_, preferred_host_device));
      tree_attn_mn_indptr_host_.push_back(
          HostMemoryVector(reserved_num_seqs + 1, dtype_aux_, preferred_host_device));
    }
    k_ragged_rope_pos_offset_host_ =
        HostMemoryVector(reserved_num_seqs, dtype_aux_, preferred_host_device);
    q_rope_position_map_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    append_position_map_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    kv_transfer_remote_position_map_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    kv_transfer_recver_id_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    kv_transfer_page_to_page_local_position_map_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    kv_transfer_page_to_page_remote_position_map_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    kv_transfer_page_to_page_recver_id_host_ =
        HostMemoryVector(prefill_chunk_size, dtype_aux_, preferred_host_device);
    cur_append_lengths_indptr_host_ =
        HostMemoryVector(reserved_num_seqs + 1, dtype_aux_, preferred_host_device);
    commit_copy_length_indptr_host_ =
        HostMemoryVector(reserved_num_seqs + 1, dtype_aux_, preferred_host_device);
    commit_copy_src_pos_in_page_table_host_ =
        HostMemoryVector(std::min(kTreeAttnMaxTreeSize * reserved_num_seqs, prefill_chunk_size),
                         dtype_aux_, preferred_host_device);
    commit_copy_dst_pos_in_page_table_host_ =
        HostMemoryVector(std::min(kTreeAttnMaxTreeSize * reserved_num_seqs, prefill_chunk_size),
                         dtype_aux_, preferred_host_device);

    for (int d = 0; d < kPagedKVCacheMaxBlockDepth; ++d) {
      if (NeedKernelBeginForward()) {
        temp_int_attn_workspace_.push_back(
            NDArray::Empty({kIntAttnWorkspaceByte / 4}, DataType::Float(32), device));
      }
      qo_indptr_on_depths_view_.push_back(NDArray());
      page_indptr_on_depths_view_.push_back(NDArray());
      page_indices_on_depths_view_.push_back(NDArray());
      length_info_on_depths_view_.push_back(NDArray());
      k_rope_pos_offset_view_.push_back(NDArray());
      tree_attn_mask_view_.push_back(NDArray());
      tree_attn_mn_indptr_view_.push_back(NDArray());
      is_chain_on_depths_.push_back(true);
    }
    // Additional workspace for the "prefill with ragged kv" kernel.
    if (NeedKernelBeginForward()) {
      temp_int_attn_workspace_.push_back(
          NDArray::Empty({kIntAttnWorkspaceByte / 4}, DataType::Float(32), device));
      temp_float_attn_workspace_ =
          NDArray::Empty({kFloatAttnWorkspaceByte / 4}, DataType::Float(32), device);
    }

    if (std::find(attn_kinds_.begin(), attn_kinds_.end(), AttnKind::kMHA) != attn_kinds_.end()) {
      temp_attn_q_device_ =
          NDArray::Empty({prefill_chunk_size_, num_qo_heads, qk_head_dim}, dtype, device);
      temp_attn_k_device_ =
          NDArray::Empty({prefill_chunk_size_, num_kv_heads, qk_head_dim}, dtype, device);
      temp_attn_v_device_ =
          NDArray::Empty({prefill_chunk_size_, num_kv_heads, v_head_dim}, dtype, device);
    }
    temp_attn_output_device_ =
        NDArray::Empty({prefill_chunk_size_, num_qo_heads, v_head_dim}, dtype, device);
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
      kv_transfer_stream_ = DeviceAPI::Get(device)->CreateStream(device);
    }

    // Create the auxiliary data manager for attention.
    // We only use the merged aux data for CUDA, since direct pointer
    // operations may have issues on other platforms.
    if (device_.device_type == DLDeviceType::kDLCUDA ||
        device_.device_type == DLDeviceType::kDLCPU) {
      aux_data_manager_ = std::make_unique<CachedPagedKVCacheAuxDataManager>(
          reserved_num_seqs, num_total_pages, prefill_chunk_size, dtype_aux_, device,
          preferred_host_device, copy_stream_);
    } else {
      aux_data_manager_ = std::make_unique<PlainPagedKVCacheAuxDataManager>(
          reserved_num_seqs, num_total_pages, prefill_chunk_size, dtype_aux_, device,
          preferred_host_device, copy_stream_);
    }

    // Right now only the "normal" RoPE mode supports the RoPE extention factors.
    if (rope_ext_factors_.defined()) {
      CHECK(rope_mode_ == RoPEMode::kNormal)
          << "The RoPE mode must be normal to support RoPE extension factors.";
    }
  }

  ~PagedAttentionKVCacheObj() {
    // Free the copy stream if defined.
    if (copy_stream_ != nullptr) {
      DeviceAPI::Get(device_)->FreeStream(device_, copy_stream_);
    }
    if (kv_transfer_stream_ != nullptr) {
      DeviceAPI::Get(device_)->FreeStream(device_, kv_transfer_stream_);
    }
  }

  /*! \brief Reset the KV cache. */
  void Clear() final {
    seq_map_.clear();
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
    seq_map_.insert({seq_id, Sequence(&global_block_pool_, block_idx)});
    dirty_aux_data_device_ = true;
  }

  void RemoveSequence(int64_t seq_id) final {
    auto it = seq_map_.find(seq_id);
    CHECK(it != seq_map_.end()) << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";
    int32_t block_idx = it->second.last_block_idx;
    // The block should have at least one reference, which comes from the sequence.
    ICHECK_GE(global_block_pool_[block_idx].external_ref_cnt, 1);
    while (block_idx != -1 && global_block_pool_[block_idx].external_ref_cnt == 1) {
      // - Free pages in the last block.
      for (int32_t page_id : global_block_pool_[block_idx].page_ids) {
        free_page_ids_.push_back(page_id);
      }
      free_block_idx_.push_back(block_idx);
      block_idx = global_block_pool_[block_idx].parent_idx;
    }
    // - Decrease the external reference of the parent block.
    if (block_idx != -1) {
      ICHECK_GT(global_block_pool_[block_idx].external_ref_cnt, 1);
      --global_block_pool_[block_idx].external_ref_cnt;
    }
    seq_map_.erase(it);
    dirty_aux_data_device_ = true;
  }

  void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id, int64_t fork_pos = -1) final {
    auto parent_it = seq_map_.find(parent_seq_id);
    CHECK(parent_it != seq_map_.end())
        << "The parent sequence \"" << parent_seq_id << "\" cannot be found in KV cache.";
    CHECK(seq_map_.find(child_seq_id) == seq_map_.end())
        << "The child sequence \"" << child_seq_id << "\" is already in the KV cache.";
    CHECK_GE(fork_pos, -1)
        << "The forked position should be non-negative, or -1 for last position as default.";
    CHECK_LE(fork_pos, parent_it->second.seq_length)
        << "The forked position should not exceed the total length of parent sequence.";
    CHECK(parent_it->second.accepted_indices_committed)
        << "The parent sequence's token tree computed in the last round of forward has not been "
           "committed with accepted nodes.";

    if (fork_pos == -1) {
      fork_pos = parent_it->second.seq_length;
    }

    if (parent_it->second.sliding_window_size != -1) {
      // If forked sequence has been enabled sliding window, check the forked position is within
      // sliding window sink size.
      const Sequence& seq = parent_it->second;
      int32_t sink_size = seq.seq_length - global_block_pool_[seq.last_block_idx].seq_length +
                          seq.last_block_attn_sink_size;
      CHECK_LE(fork_pos, sink_size)
          << "The parent sequence \"" << parent_seq_id
          << "\" is enabled with sliding window and thus only can be forked within sink size = "
          << sink_size << ". But the forked position = " << fork_pos << ".";
    }

    if (fork_pos == parent_it->second.seq_length && fork_pos % page_size_ == 0 &&
        global_block_pool_[parent_it->second.last_block_idx].seq_length > 0) {
      // To enable the parent sequence to continue decode after the fork,
      // we add a new empty block at the end of the parent sequence.
      // So the new decoded KV data will go into the new block.
      int32_t new_block_idx = GetFreeBlock();
      global_block_pool_[new_block_idx].start_pos = parent_it->second.seq_length;
      global_block_pool_[new_block_idx].parent_idx = parent_it->second.last_block_idx;
      global_block_pool_[new_block_idx].external_ref_cnt = 1;
      parent_it->second.last_block_idx = new_block_idx;
    }

    int32_t child_block_idx = GetFreeBlock();
    std::vector<int32_t> trace = parent_it->second.GetBlockTrace(global_block_pool_);
    int64_t in_block_offset = fork_pos;
    for (int32_t forked_block_idx : trace) {
      if (forked_block_idx != trace.back()) {
        CHECK_GT(global_block_pool_[forked_block_idx].seq_length, 0);
        CHECK_EQ(global_block_pool_[forked_block_idx].seq_length % page_size_, 0);
        if (global_block_pool_[forked_block_idx].seq_length <= in_block_offset) {
          in_block_offset -= global_block_pool_[forked_block_idx].seq_length;
          continue;
        }
      }
      int32_t in_page_offset = in_block_offset % page_size_;
      int32_t moved_offset = in_block_offset - in_page_offset;
      int32_t moved_pages = moved_offset / page_size_;
      if (moved_pages == 0) {
        // Forked at the first page in block
        int32_t parent_block_idx = global_block_pool_[forked_block_idx].parent_idx;
        if (parent_block_idx != -1) {
          ++global_block_pool_[parent_block_idx].external_ref_cnt;
        }
        // Update child block start position and parent index
        global_block_pool_[child_block_idx].parent_idx = parent_block_idx;
      } else {
        // Forked at the second or latter page in block
        int32_t parent_block_idx = GetFreeBlock();
        // Insert new parent block before forked block and link child block
        global_block_pool_[parent_block_idx].parent_idx =
            global_block_pool_[forked_block_idx].parent_idx;
        global_block_pool_[forked_block_idx].parent_idx = parent_block_idx;
        global_block_pool_[child_block_idx].parent_idx = parent_block_idx;
        global_block_pool_[parent_block_idx].external_ref_cnt = 2;

        // Move common leading pages to new parent block
        auto first_page = global_block_pool_[forked_block_idx].page_ids.begin();
        auto last_page = global_block_pool_[forked_block_idx].page_ids.begin() + moved_pages;
        global_block_pool_[parent_block_idx].page_ids = {first_page, last_page};
        global_block_pool_[forked_block_idx].page_ids.erase(first_page, last_page);

        // Update start position per blocks
        global_block_pool_[parent_block_idx].start_pos =
            global_block_pool_[forked_block_idx].start_pos;
        global_block_pool_[forked_block_idx].start_pos += moved_offset;

        // Update in-block sequence length per blocks
        global_block_pool_[parent_block_idx].seq_length = moved_offset;
        global_block_pool_[forked_block_idx].seq_length -= moved_offset;

        // Update sliding window sink size if sliding window is enabled and the forked block is the
        // last block
        if (parent_it->second.sliding_window_size != -1 &&
            forked_block_idx == parent_it->second.last_block_idx) {
          CHECK_LE(moved_offset, parent_it->second.last_block_attn_sink_size);
          parent_it->second.last_block_attn_sink_size -= moved_offset;
        }
      }
      global_block_pool_[child_block_idx].start_pos = fork_pos - in_page_offset;
      global_block_pool_[child_block_idx].seq_length = in_page_offset;

      if (in_page_offset > 0) {
        // Fork within a page and copy common page to child block partially
        int32_t src_page_id = global_block_pool_[forked_block_idx].page_ids[0];
        int32_t tgt_page_id = GetFreePage();
        global_block_pool_[child_block_idx].page_ids.push_back(tgt_page_id);
        CopySinglePage(src_page_id, tgt_page_id, in_page_offset);
      }
      break;
    }
    // Create the child sequence with the child block.
    seq_map_.insert({child_seq_id, Sequence(&global_block_pool_, child_block_idx)});
    dirty_aux_data_device_ = true;
  }

  void CopySinglePage(int32_t src_page_id, int32_t tgt_page_id, int64_t copy_length) {
    if (copy_stream_ != compute_stream_) {
      // Set the copy stream for copy.
      DeviceAPI::Get(device_)->SetStream(device_, copy_stream_);
    }
    for (int layer = 0; layer < num_layers_; ++layer) {
      NDArray page_layer_view = pages_[layer];
      f_copy_single_page_(page_layer_view, src_page_id, tgt_page_id, copy_length);
    }
    if (copy_stream_ != compute_stream_) {
      // Set the compute stream back.
      DeviceAPI::Get(device_)->SetStream(device_, compute_stream_);
    }
  }

  void CompactKVCopy() {
    int total_copy_length = commit_copy_length_indptr_host_.back();
    ICHECK_GE(total_copy_length, 0);
    if (total_copy_length == 0) {
      return;
    }

    // Copy indptr/src/dst arrays to GPU.
    aux_data_manager_->ResetCompactKVAuxDataCopy();
    NDArray commit_copy_length_indptr_view =
        aux_data_manager_->CopyCommitLengthIndptrAsync(&commit_copy_length_indptr_host_);
    NDArray commit_copy_src_dst_pos_in_page_table_view =
        aux_data_manager_->CopyCommitSrcDstPosInPageTableAsync(
            &commit_copy_src_pos_in_page_table_host_, &commit_copy_dst_pos_in_page_table_host_);
    aux_data_manager_->CommitCompactKVAuxDataCopy();

    // Invoke the copy kernel on copy stream.
    if (copy_stream_ != compute_stream_) {
      // Set the copy stream for copy.
      DeviceAPI::Get(device_)->SetStream(device_, copy_stream_);
    }
    ICHECK(f_compact_copy_.defined()) << "Function \"f_compact_copy\" is not defined.";
    for (int layer = 0; layer < num_layers_; ++layer) {
      f_compact_copy_(pages_[layer], commit_copy_length_indptr_view,
                      commit_copy_src_dst_pos_in_page_table_view, cur_batch_size_);
    }
    if (copy_stream_ != compute_stream_) {
      // Set the compute stream back.
      DeviceAPI::Get(device_)->SetStream(device_, compute_stream_);
    }

    // Note: We do not explicitly synchronize the copy stream here.
    // The safety is guaranteed by the synchronization pushed by the next round
    // of BeginForward, which also copies auxiliary data structure to GPU.
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
    const Block& last_block = global_block_pool_[it->second.last_block_idx];
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

    CHECK_GE(n, 0) << "The length of popping " << n << " cannot be negative.";
    CHECK_LE(n, it->second.seq_length)
        << "The sequence only has length " << it->second.seq_length
        << ", while the length of pop is " << n << " which exceeds the whole sequence length.";
    if (n == 0) {
      return;
    }

    int32_t block_idx = it->second.last_block_idx;
    // The block should have at least one reference, which comes from the sequence.
    ICHECK_GE(global_block_pool_[block_idx].external_ref_cnt, 1);
    while (block_idx != -1 && global_block_pool_[block_idx].external_ref_cnt == 1) {
      if (n > global_block_pool_[block_idx].seq_length) {
        n -= global_block_pool_[block_idx].seq_length;
        it->second.seq_length -= global_block_pool_[block_idx].seq_length;
        for (int32_t page_id : global_block_pool_[block_idx].page_ids) {
          free_page_ids_.push_back(page_id);
        }
        free_block_idx_.push_back(block_idx);
        block_idx = global_block_pool_[block_idx].parent_idx;
        it->second.last_block_idx = block_idx;
        continue;
      }
      if (n <= global_block_pool_[block_idx].seq_length) {
        int64_t cur_npage = global_block_pool_[block_idx].page_ids.size();
        int64_t tgt_npage =
            (global_block_pool_[block_idx].seq_length - n + page_size_ - 1) / page_size_;
        while (cur_npage > tgt_npage) {
          free_page_ids_.push_back(global_block_pool_[block_idx].page_ids.back());
          global_block_pool_[block_idx].page_ids.pop_back();
          --cur_npage;
        }
        it->second.seq_length -= n;
        global_block_pool_[block_idx].seq_length -= n;
        n = 0;
        break;
      }
    }

    if (n) {
      // We use a temporary sequence id for fork.
      // This temporary seq id will immediately end its effect outside this function.
      int64_t temp_seq_id = -1 - seq_id;
      CHECK(seq_map_.find(temp_seq_id) == seq_map_.end());
      ForkSequence(seq_id, temp_seq_id, it->second.seq_length - n);
      CHECK(seq_map_.find(temp_seq_id) != seq_map_.end());
      RemoveSequence(seq_id);
      CHECK(seq_map_.find(seq_id) == seq_map_.end());
      auto it = seq_map_.find(temp_seq_id);
      seq_map_.insert({seq_id, it->second});
      seq_map_.erase(temp_seq_id);
    }

    dirty_aux_data_device_ = true;
  }

  /************** Raw Info Query **************/

  bool Empty() const final {
    return seq_map_.empty() &&                                     //
           free_block_idx_.size() == global_block_pool_.size() &&  //
           free_page_ids_.size() == static_cast<size_t>(num_total_pages_);
  }

  int32_t GetNumAvailablePages() const final { return free_page_ids_.size(); }

  int32_t GetTotalSequenceLength() const final {
    int32_t total_seq_len = 0;
    for (const auto& it : seq_map_) {
      total_seq_len += it.second.seq_length;
    }
    return total_seq_len;
  }

  /************** Attention **************/

  void BeginForward(const IntTuple& seq_ids, const IntTuple& append_lengths,
                    const Optional<IntTuple>& opt_token_tree_parent_ptr) final {
    // Note: MLA does not supported tree attention for now.
    if (attn_kinds_[0] == AttnKind::kMLA) {
      CHECK(!opt_token_tree_parent_ptr.defined()) << "Tree attention is not supported yet for MLA";
    }

    CHECK_EQ(seq_ids.size(), append_lengths.size())
        << "The seq_ids size (" << seq_ids.size() << ") and append_lengths size ("
        << append_lengths.size() << ") mismatch.";
    cur_batch_size_ = seq_ids.size();
    cur_seq_ids_ = seq_ids;
    cur_append_lengths_ = append_lengths;

    // - Collect sequence/block/page information for attention.
    std::vector<Sequence*> sequences;
    std::vector<int32_t> last_block_length_before_append;
    is_decode_request_ = true;
    sequences.reserve(cur_batch_size_);
    last_block_length_before_append.reserve(cur_batch_size_);
    k_ragged_rope_pos_offset_host_.clear();
    for (int i = 0; i < cur_batch_size_; ++i) {
      auto it = seq_map_.find(seq_ids[i]);
      CHECK(it != seq_map_.end()) << "The sequence \"" << seq_ids[i]
                                  << "\" cannot be found in KV cache.";
      sequences.push_back(&it->second);
      last_block_length_before_append.push_back(
          global_block_pool_[it->second.last_block_idx].seq_length);
      int k_rope_offset = it->second.seq_length;
      if (!it->second.accepted_indices_committed) {
        int tree_size = static_cast<int>(it->second.token_tree_parent_ptr.size());
        k_rope_offset -= tree_size;
      }
      k_ragged_rope_pos_offset_host_.push_back(k_rope_offset);
      it->second.seq_length += append_lengths[i];
      if (append_lengths[i] != 1) {
        is_decode_request_ = false;
      }
    }

    auto [block_ids_on_depths, trailing_blocks] = GetBlockIdsOnDepth(sequences);
    num_depths_ =
        std::min(static_cast<int>(block_ids_on_depths.size()), kPagedKVCacheMaxBlockDepth);
    ICHECK_LE(num_depths_, kPagedKVCacheMaxBlockDepth);

    std::vector<std::vector<std::pair<int32_t, int32_t>>> chunked_block_ids_arr;
    chunked_block_ids_arr.reserve(num_depths_);
    use_decode_kernel_.clear();
    for (int d = 0; d < num_depths_; ++d) {
      // We force the blocks at maximum depth not to coalesce, so that it can be concatenated with
      // trailing exceeding blocks.
      auto [chunked_block_ids, use_decode_kernel] = GetChunkedBlockIds(
          block_ids_on_depths[d], /*enable_coalesce=*/d != kPagedKVCacheMaxBlockDepth - 1);
      chunked_block_ids_arr.push_back(chunked_block_ids);
      use_decode_kernel_.push_back(use_decode_kernel);
    }

    if (num_depths_ == kPagedKVCacheMaxBlockDepth) {
      // Since we force the blocks at maximum depth not to coalesce, the output blocks at maximum
      // depth must have the same size as current batch.
      CHECK_EQ(chunked_block_ids_arr[num_depths_ - 1].size(), cur_batch_size_);
    }

    append_before_attn_ = !support_sliding_window_ && use_decode_kernel_.back();
    if (NeedKernelBeginForward() && num_qo_heads_ / num_kv_heads_ >= 4) {
      // When GQA group size is at least 4 and FlashInfer is enabled,
      // we always use prefill kernel for better performance.
      std::fill(use_decode_kernel_.begin(), use_decode_kernel_.end(), /*value=*/false);
    }

    bool has_previous_tree =
        std::any_of(sequences.begin(), sequences.end(),
                    [](const Sequence* sequence) { return !sequence->accepted_indices_committed; });
    if (has_previous_tree) {
      append_before_attn_ = true;
    }

    // - Check token tree validity and process the token tree.
    if (opt_token_tree_parent_ptr.defined()) {
      CHECK(!support_sliding_window_) << "Tree attention does not support sliding window.";
      CHECK(rope_mode_ != RoPEMode::kInline) << "Tree attention does not support inline RoPE mode.";
      ConstructTokenTreeMask(sequences, opt_token_tree_parent_ptr.value(), block_ids_on_depths,
                             trailing_blocks);
    } else {
      // The input batch does not form trees. So each sequence in the batch
      // is required to have all past accepted tokens committed.
      for (int i = 0; i < cur_batch_size_; ++i) {
        Sequence* sequence = sequences[i];
        CHECK(sequence->accepted_indices_committed)
            << "The input batch does not form a tree, in which case the sequences in the input "
               "batch are expected to have their accepted tokens token tree nodes committed. "
               "Please invoke CommitAcceptedTokenTreeNodes for sequence "
            << seq_ids[i];
        sequence->is_chain = true;
        sequence->token_tree_parent_ptr.clear();
        sequence->token_tree_node_depths.clear();
      }
      std::fill(is_chain_on_depths_.begin(), is_chain_on_depths_.end(), true);
    }

    if (append_before_attn_) {
      // Right now we use different kernels when depth is 1 or not 1.
      // For the case where maximum depth is 1, we create the auxiliary
      // data structure with regard to the page table after appending.
      for (int i = 0; i < cur_batch_size_; ++i) {
        ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);
      }
    }

    for (int d = 0; d < num_depths_; ++d) {
      HostMemoryVector& qo_indptr_h = qo_indptr_on_depths_host_[d];
      HostMemoryVector& page_indptr_h = page_indptr_on_depths_host_[d];
      HostMemoryVector& page_indices_h = page_indices_on_depths_host_[d];
      HostMemoryVector& last_page_len_h = last_page_len_on_depths_host_[d];
      HostMemoryVector& sliding_window_offset_h = sliding_window_offset_on_depths_host_[d];
      HostMemoryVector& sink_size_h = sink_size_on_depths_host_[d];
      HostMemoryVector& k_rope_pos_offset_h = k_rope_pos_offset_on_depths_host_[d];
      qo_indptr_h.clear();
      page_indptr_h.clear();
      page_indices_h.clear();
      last_page_len_h.clear();
      sliding_window_offset_h.clear();
      sink_size_h.clear();
      k_rope_pos_offset_h.clear();
      qo_indptr_h.push_back(0);
      page_indptr_h.push_back(0);
      for (int i = 0; i < static_cast<int>(chunked_block_ids_arr[d].size()); ++i) {
        const auto& [block_id, chunk_append_length] = chunked_block_ids_arr[d][i];
        qo_indptr_h.push_back(qo_indptr_h.back() + chunk_append_length);
        if (block_id == -1) {
          page_indptr_h.push_back(page_indptr_h.back());
          last_page_len_h.push_back(0);
          sliding_window_offset_h.push_back(0);
          sink_size_h.push_back(0);
          k_rope_pos_offset_h.push_back(0);
        } else {
          if (d < kPagedKVCacheMaxBlockDepth - 1) {
            // Blocks not at maximum depth
            const Block& block = global_block_pool_[block_id];
            page_indptr_h.push_back(page_indptr_h.back() + block.page_ids.size());
            for (int32_t page_id : block.page_ids) {
              page_indices_h.push_back(page_id);
            }
            last_page_len_h.push_back(
                block.seq_length == 0
                    ? 0
                    : (block.seq_length - block.sink_length + block.sliding_window_offset - 1) %
                              page_size_ +
                          1);
            sliding_window_offset_h.push_back(block.sliding_window_offset);
            sink_size_h.push_back(block.sink_length);
            k_rope_pos_offset_h.push_back(block.start_pos);
          } else {
            // Blocks at maximum depth
            const Block& block = global_block_pool_[block_id];
            int32_t num_pages = static_cast<int32_t>(block.page_ids.size());
            int32_t total_seq_length = static_cast<int32_t>(block.seq_length);
            int32_t last_block_id = block_id;
            for (int32_t page_id : block.page_ids) {
              page_indices_h.push_back(page_id);
            }
            for (int32_t id : trailing_blocks[i]) {
              // Collect trailing blocks if available
              const Block& block = global_block_pool_[id];
              for (int32_t page_id : block.page_ids) {
                page_indices_h.push_back(page_id);
              }
              num_pages += block.page_ids.size();
              total_seq_length += block.seq_length;
              last_block_id = id;
            }
            page_indptr_h.push_back(page_indptr_h.back() + num_pages);
            const Block& last_block = global_block_pool_[last_block_id];
            last_page_len_h.push_back(total_seq_length == 0
                                          ? 0
                                          : (total_seq_length - last_block.sink_length +
                                             last_block.sliding_window_offset - 1) %
                                                    page_size_ +
                                                1);
            sliding_window_offset_h.push_back(last_block.sliding_window_offset);
            sink_size_h.push_back(last_block.sink_length);
            k_rope_pos_offset_h.push_back(block.start_pos);
          }
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
    kv_transfer_remote_position_map_host_.clear();
    kv_transfer_recver_id_host_.clear();
    kv_transfer_page_to_page_local_position_map_host_.clear();
    kv_transfer_page_to_page_remote_position_map_host_.clear();
    kv_transfer_page_to_page_recver_id_host_.clear();
    transfer_kv_ = false;
    page_to_page_transfer_kv_ = false;
    for (int i = 0; i < cur_batch_size_; ++i) {
      int64_t append_length = append_lengths[i];
      const Block& block = global_block_pool_[sequences[i]->last_block_idx];
      for (int64_t pos = 0; pos < append_length; ++pos) {
        if (sequences[i]->token_tree_node_depths.empty()) {
          q_rope_position_map_host_.push_back(k_ragged_rope_pos_offset_host_[i] + pos);
        } else {
          int64_t offset_in_tree =
              static_cast<int64_t>(sequences[i]->token_tree_parent_ptr.size()) - append_length;
          ICHECK_GE(offset_in_tree, 0);
          q_rope_position_map_host_.push_back(
              k_ragged_rope_pos_offset_host_[i] +
              sequences[i]->token_tree_node_depths[offset_in_tree + pos]);
        }

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
        int64_t pos_in_seq = sequences[i]->seq_length - append_length + pos;
        int64_t seq_send_start = sequences[i]->kv_transfer_metadata.start;
        if (pos_in_seq < seq_send_start) {
          kv_transfer_remote_position_map_host_.push_back(-1);
          kv_transfer_recver_id_host_.push_back(-1);
        } else {
          transfer_kv_ = true;
          kv_transfer_remote_position_map_host_.push_back(
              sequences[i]->kv_transfer_metadata.remote_position_map[pos_in_seq - seq_send_start]);
          kv_transfer_recver_id_host_.push_back(
              sequences[i]->kv_transfer_metadata.recver_pe_offset);
        }
      }
      if (!sequences[i]->kv_transfer_metadata.local_position_map.empty()) {
        page_to_page_transfer_kv_ = true;
        for (int pos = 0;
             pos < static_cast<int>(sequences[i]->kv_transfer_metadata.local_position_map.size());
             ++pos) {
          kv_transfer_page_to_page_local_position_map_host_.push_back(
              sequences[i]->kv_transfer_metadata.local_position_map[pos]);
          kv_transfer_page_to_page_remote_position_map_host_.push_back(
              sequences[i]->kv_transfer_metadata.remote_position_map[pos]);
          kv_transfer_page_to_page_recver_id_host_.push_back(
              sequences[i]->kv_transfer_metadata.recver_pe_offset);
        }
        sequences[i]->kv_transfer_metadata.local_position_map.clear();
      }
    }
  }

  void EndForward() final {
    if (kv_transfer_stream_ != nullptr) {
      DeviceAPI::Get(device_)->SyncStreamFromTo(device_, kv_transfer_stream_, compute_stream_);
    }
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

  IntTuple DisaggPrepareRecv(int64_t seq_id, int append_length) final {
    // No CPU to GPU copy is needed.
    // Essentially we
    // (step 1.) redirect the preparation to BeginForward.
    BeginForward({seq_id}, {append_length}, /*opt_token_tree_parent_ptr=*/NullOpt);
    // (step 2.) fetch the append_position_map, compress and return.
    // Compression format: [n, begin_1, length_1, begin_2, length_2, ..., begin_n, length_n]
    // The compressed format will be decompressed to:
    // [begin_1, begin_1+1, ..., begin_1+length_1-1, ..., begin_n, ..., begin_n+length_n-1]
    CHECK_EQ(append_position_map_host_.size(), append_length);
    std::vector<int64_t> compressed_append_pos_map{/*num_segments=*/1,
                                                   append_position_map_host_[0]};
    for (int i = 1; i < append_length; ++i) {
      if (append_position_map_host_[i] != append_position_map_host_[i - 1] + 1) {
        // Terminate the current segment.
        compressed_append_pos_map.push_back(append_position_map_host_[i - 1] -
                                            compressed_append_pos_map.back() + 1);
        // Start a new segment.
        ++compressed_append_pos_map[0];
        compressed_append_pos_map.push_back(append_position_map_host_[i]);
      }
    }
    // Terminate the last segment.
    compressed_append_pos_map.push_back(append_position_map_host_.back() -
                                        compressed_append_pos_map.back() + 1);
    // The compressed array size should be "num_segments * 2 + 1".
    CHECK_EQ(compressed_append_pos_map.size(), compressed_append_pos_map[0] * 2 + 1);
    return IntTuple{compressed_append_pos_map};
  }

  void DisaggMarkSend(int64_t seq_id, int64_t begin, const IntTuple& compressed_remote_position_map,
                      int32_t recver_pe_offset) {
    ICHECK(f_transfer_kv_.defined());
    auto it = seq_map_.find(seq_id);
    CHECK(it != seq_map_.end()) << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";
    Sequence* sequence = &it->second;
    sequence->kv_transfer_metadata.start = begin;
    int nsegments = compressed_remote_position_map[0];
    sequence->kv_transfer_metadata.remote_position_map.clear();
    for (int i = 0; i < nsegments; ++i) {
      int begin = compressed_remote_position_map[2 * i + 1];
      int length = compressed_remote_position_map[2 * i + 2];
      for (int j = 0; j < length; ++j) {
        sequence->kv_transfer_metadata.remote_position_map.push_back(begin + j);
      }
    }
    sequence->kv_transfer_metadata.recver_pe_offset = recver_pe_offset;

    sequence->kv_transfer_metadata.local_position_map.clear();
    if (begin >= sequence->seq_length) {
      return;
    }
    // Need to send existing KV.
    CHECK_GT(static_cast<int>(sequence->kv_transfer_metadata.remote_position_map.size()),
             sequence->seq_length - begin)
        << "Need at least one token to prefill";
    std::vector<int32_t> trace = sequence->GetBlockTrace(global_block_pool_);
    sequence->kv_transfer_metadata.local_position_map.reserve(sequence->seq_length - begin);
    bool done = false;
    for (auto it_block_id = trace.rbegin(); it_block_id != trace.rend(); ++it_block_id) {
      const Block& block = global_block_pool_[*it_block_id];
      for (int i = block.seq_length - 1; i >= 0; --i) {
        int32_t offset =
            i < block.sink_length ? i : i - block.sink_length + block.sliding_window_offset;
        int page_id = block.page_ids[offset / page_size_];
        int page_offset = offset % page_size_;
        sequence->kv_transfer_metadata.local_position_map.push_back(page_id * page_size_ +
                                                                    page_offset);
        if (static_cast<int>(sequence->kv_transfer_metadata.local_position_map.size()) ==
            sequence->seq_length - begin) {
          done = true;
          break;
        }
      }
      if (done) {
        break;
      }
    }
    std::reverse(sequence->kv_transfer_metadata.local_position_map.begin(),
                 sequence->kv_transfer_metadata.local_position_map.end());
  }

  void AttentionWithFusedQKV(int64_t layer_id, NDArray qkv_data, Optional<NDArray> mask,
                             NDArray o_data, double attn_score_scaling_factor) final {
    // Part 1. Shape and dtype check.
    int64_t local_layer_id = layer_id - layer_id_begin_offset_;
    CHECK_GE(local_layer_id, 0);
    CHECK_LT(local_layer_id, num_layers_);
    NDArray pages = pages_[local_layer_id];
    CHECK(qkv_data.DataType() == pages.DataType());
    CHECK(o_data.DataType() == pages.DataType());
    CHECK(attn_kinds_[layer_id] == AttnKind::kMHA);

    // qkv_data: (num_total_length, num_qo_heads + 2 * num_kv_heads, qk_head_dim)
    // o_data: (num_total_length, num_qo_heads, qk_head_dim)

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

    CHECK_EQ(qkv_data->shape[2], qk_head_dim_);
    int64_t total_seq_length = 0;
    for (int64_t seq_id = 0; seq_id < cur_batch_size_; ++seq_id) {
      total_seq_length += cur_append_lengths_[seq_id];
    }
    CHECK_LE(total_seq_length, qkv_data->shape[0]);
    // Sync the copy stream and the compute stream.
    ComputeStreamWaitForCopyStream();
    // The auxiliary data structure on device must have been synchronized.
    ICHECK(!dirty_aux_data_device_);

    NDArray q_data = temp_attn_q_device_.CreateView({total_seq_length, num_qo_heads_, qk_head_dim_},
                                                    qkv_data->dtype);
    NDArray k_data = temp_attn_k_device_.CreateView({total_seq_length, num_kv_heads_, qk_head_dim_},
                                                    qkv_data->dtype);
    NDArray v_data = temp_attn_v_device_.CreateView({total_seq_length, num_kv_heads_, qk_head_dim_},
                                                    qkv_data->dtype);

    NDArray qkv_data_view = qkv_data;
    NDArray o_data_view = o_data;
    if (total_seq_length != qkv_data->shape[0]) {
      qkv_data_view = qkv_data.CreateView(
          {total_seq_length, qkv_data->shape[1], qkv_data->shape[2]}, qkv_data->dtype);
      o_data_view =
          o_data.CreateView({total_seq_length, num_qo_heads_, qk_head_dim_}, qkv_data->dtype);
    }
    // Part 2. Split fused qkv and apply rotary embedding to q/k data.
    if (transfer_kv_) {
      // The the compute stream needs to wait for the KV transfer stream.
      DeviceAPI::Get(device_)->SyncStreamFromTo(device_, kv_transfer_stream_, compute_stream_);
    }
    if (!rope_ext_factors_.defined()) {
      f_split_rotary_(qkv_data_view, q_rope_position_map_view_, q_data, k_data, v_data,
                      static_cast<int>(rope_mode_ == RoPEMode::kNormal));
    } else {
      f_split_rotary_(qkv_data_view, q_rope_position_map_view_, q_data, k_data, v_data,
                      rope_ext_factors_.value());
    }

    // Part 3. Append k/v data to kv-cache if flag "append_before_attn" is set.
    if (append_before_attn_) {
      f_transpose_append_(pages_[local_layer_id], k_data, v_data, append_position_map_view_);
    }
    // Part 4: KV transfer
    if (page_to_page_transfer_kv_) {
      DeviceAPI::Get(device_)->SyncStreamFromTo(device_, copy_stream_, kv_transfer_stream_);
      // FIXME: if the sender and recver's PP/TP degree do not match, we will need to first
      // get the view of remote pages, and then take the specific remote layer.
      // The KV transfer stream nees to wait for the compute stream.
      f_transfer_kv_page_to_page_.value()(pages_[local_layer_id], pages_[local_layer_id],
                                          kv_transfer_page_to_page_remote_position_map_view_,
                                          kv_transfer_page_to_page_local_position_map_view_,
                                          kv_transfer_page_to_page_recver_id_view_,
                                          kv_transfer_stream_);
    }
    if (transfer_kv_) {
      // FIXME: if the sender and recver's PP/TP degree do not match, we will need to first
      // get the view of remote pages, and then take the specific remote layer.
      // The KV transfer stream nees to wait for the compute stream.
      DeviceAPI::Get(device_)->SyncStreamFromTo(device_, compute_stream_, kv_transfer_stream_);
      f_transfer_kv_.value()(pages_[local_layer_id], k_data, v_data,
                             kv_transfer_remote_position_map_view_, kv_transfer_recver_id_view_,
                             kv_transfer_stream_);
    }
    // Part 5: perform attention
    AttentionInternal(layer_id, q_data, k_data, v_data, o_data_view, attn_score_scaling_factor);
    // Part 6. Append k/v data to kv-cache if flag "append_before_attn" is not set.
    if (!append_before_attn_) {
      f_transpose_append_(pages_[local_layer_id], k_data, v_data, append_position_map_view_);
    }
  }

  void MLAAbsorbed(int64_t layer_id, NDArray q_data, NDArray compressed_kv_data, NDArray k_pe_data,
                   NDArray o_data, double attn_score_scaling_factor) {
    // Part 1. Shape and dtype check.
    int64_t local_layer_id = layer_id - layer_id_begin_offset_;
    CHECK_GE(local_layer_id, 0);
    CHECK_LT(local_layer_id, num_layers_);
    NDArray pages = pages_[local_layer_id];
    CHECK(q_data.DataType() == pages.DataType());
    CHECK(compressed_kv_data.DataType() == pages.DataType());
    CHECK(k_pe_data.DataType() == pages.DataType());
    CHECK(o_data.DataType() == pages.DataType());
    CHECK(attn_kinds_[layer_id] == AttnKind::kMLA);

    // q_data: (num_total_length, num_qo_heads, qk_head_dim)
    // compressed_kv_data: (num_total_length, qk_head_dim - qk_rope_head_dim)
    // k_pe_data: (num_total_length, qk_rope_head_dim)
    // o_data: (num_total_length, num_qo_heads, v_head_dim)
    CHECK_EQ(q_data->ndim, 3);
    CHECK_EQ(compressed_kv_data->ndim, 2);
    CHECK_EQ(k_pe_data->ndim, 2);
    CHECK_EQ(o_data->ndim, 3);

    int64_t total_seq_length = 0;
    for (int64_t seq_id = 0; seq_id < cur_batch_size_; ++seq_id) {
      total_seq_length += cur_append_lengths_[seq_id];
    }
    CHECK_LE(q_data->shape[0], total_seq_length);
    CHECK_LE(compressed_kv_data->shape[0], total_seq_length);
    CHECK_LE(k_pe_data->shape[0], total_seq_length);
    CHECK_LE(o_data->shape[0], total_seq_length);
    CHECK_EQ(q_data->shape[1], num_qo_heads_);
    CHECK_EQ(o_data->shape[1], num_qo_heads_);
    CHECK_EQ(q_data->shape[2], qk_head_dim_);
    CHECK_EQ(compressed_kv_data->shape[1], qk_head_dim_ - qk_rope_head_dim_);
    CHECK_EQ(k_pe_data->shape[1], qk_rope_head_dim_);
    CHECK_EQ(o_data->shape[2], v_head_dim_);

    // Sync the copy stream and the compute stream.
    ComputeStreamWaitForCopyStream();
    // The auxiliary data structure on device must have been synchronized.
    ICHECK(!dirty_aux_data_device_);

    // Append k/v data to kv-cache if flag "append_before_attn" is set.
    if (append_before_attn_) {
      f_transpose_append_mla_(pages_[local_layer_id], compressed_kv_data, k_pe_data,
                              append_position_map_view_);
    }
    // Perform MLA with weight absorption.
    MLAAbsorbedInternal(layer_id, q_data, compressed_kv_data, k_pe_data, o_data,
                        attn_score_scaling_factor);
    // Append k/v data to kv-cache if flag "append_before_attn" is not set.
    if (!append_before_attn_) {
      f_transpose_append_mla_(pages_[local_layer_id], compressed_kv_data, k_pe_data,
                              append_position_map_view_);
    }
  }

  void MLANormal(int64_t layer_id, NDArray q_data, NDArray k_data, NDArray v_data,
                 NDArray compressed_kv_data, NDArray k_pe_data, NDArray o_data,
                 double attn_score_scaling_factor) {
    // Todo(ruihang): implement it
  }

  void LinearAttention(int64_t layer_id, NDArray q_data, NDArray k_data, NDArray v_data,
                       double attn_score_scaling_factor) {
    // Todo(ruihang): implement it
  }

  void CommitAcceptedTokenTreeNodes(const IntTuple& seq_ids, const IntTuple& leaf_indices) final {
    CHECK_EQ(seq_ids.size(), leaf_indices.size())
        << "The given seq_ids and leaf_indices have different size.";
    int num_seq_to_commit = seq_ids.size();

    std::vector<Sequence*> sequences;
    sequences.reserve(num_seq_to_commit);
    bool is_chain = true;
    for (int i = 0; i < num_seq_to_commit; ++i) {
      auto it = seq_map_.find(seq_ids[i]);
      CHECK(it != seq_map_.end()) << "The sequence \"" << seq_ids[i]
                                  << "\" cannot be found in KV cache.";
      sequences.push_back(&it->second);
      is_chain = it->second.is_chain;
      CHECK(leaf_indices[i] == -1 || !it->second.accepted_indices_committed)
          << "The accepted nodes of sequence " << seq_ids[i] << " are already committed.";
      CHECK_GE(leaf_indices[i], -1)
          << "Invalid tree index " << leaf_indices[i] << " which is less than -1";
      CHECK_LT(leaf_indices[i], static_cast<int64_t>(it->second.token_tree_parent_ptr.size()))
          << "Invalid tree index " << leaf_indices[i]
          << " which is larger than or equals to the append length "
          << it->second.token_tree_parent_ptr.size() << " of the sequence";
    }

    if (!is_chain) {
      commit_copy_length_indptr_host_.clear();
      commit_copy_src_pos_in_page_table_host_.clear();
      commit_copy_dst_pos_in_page_table_host_.clear();
      commit_copy_length_indptr_host_.push_back(0);

      for (int i = 0; i < num_seq_to_commit; ++i) {
        if (leaf_indices[i] == -1) {
          // No node is accepted. All nodes in the token tree need to be popped.
          commit_copy_length_indptr_host_.push_back(commit_copy_length_indptr_host_.back());
          continue;
        }

        // Get the accepted node path on the token tree.
        std::vector<int32_t> path_on_tree;
        path_on_tree.reserve(sequences[i]->token_tree_node_depths[leaf_indices[i]] + 1);
        int node = leaf_indices[i];
        while (node != -1) {
          path_on_tree.push_back(node);
          node = sequences[i]->token_tree_parent_ptr[node];
        }
        ICHECK_EQ(path_on_tree.size(), sequences[i]->token_tree_node_depths[leaf_indices[i]] + 1);
        // Get the destination array (range [0, path_length - 1)) of KV cache copy.
        std::vector<int32_t> copy_dst_pos_in_seq;
        copy_dst_pos_in_seq.resize(path_on_tree.size());
        std::iota(copy_dst_pos_in_seq.rbegin(), copy_dst_pos_in_seq.rend(), /*value=*/0);
        // Remove the positions whose KV data do not need copy.
        while (!path_on_tree.empty() && path_on_tree.back() == copy_dst_pos_in_seq.back()) {
          path_on_tree.pop_back();
          copy_dst_pos_in_seq.pop_back();
        }
        // Reverse the position arrays so that they are in ascending order.
        std::reverse(path_on_tree.begin(), path_on_tree.end());
        std::reverse(copy_dst_pos_in_seq.begin(), copy_dst_pos_in_seq.end());

        // Convert the in-sequence src/dst positions to src/dst positions in page table
        // by looking up "append_position_map".
        for (int p = 0; p < static_cast<int>(path_on_tree.size()); ++p) {
          commit_copy_src_pos_in_page_table_host_.push_back(
              append_position_map_host_[cur_append_lengths_indptr_host_[i] + path_on_tree[p]]);
          commit_copy_dst_pos_in_page_table_host_.push_back(
              append_position_map_host_[cur_append_lengths_indptr_host_[i] +
                                        copy_dst_pos_in_seq[p]]);
        }
        commit_copy_length_indptr_host_.push_back(commit_copy_length_indptr_host_.back() +
                                                  path_on_tree.size());
      }

      // Compact the KV data for each sequence by copying KV data.
      CompactKVCopy();
    }

    // - Update the KV cache page data structure.
    //   Note: Function "PopN" only changes the page table structure and does not
    //         change the KV cache data. Therefore, we can directly use it, since
    //         we have already launched all copies.
    for (int i = 0; i < num_seq_to_commit; ++i) {
      int64_t length_to_pop =
          cur_append_lengths_[i] -
          (leaf_indices[i] != -1 ? (sequences[i]->token_tree_node_depths[leaf_indices[i]] + 1) : 0);
      PopN(cur_seq_ids_[i], length_to_pop);
      // Reset the sequence states.
      sequences[i]->accepted_indices_committed = true;
      sequences[i]->token_tree_parent_ptr.clear();
      sequences[i]->token_tree_node_depths.clear();
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

    // k/v_data: (num_layers, seq_length, num_kv_heads, qk_head_dim)
    static constexpr const char* error_msg =
        "DebugGetKV expects the k_data in layout (num_layers, seq_length, num_kv_heads, "
        "qk_head_dim).";
    std::vector<NDArray*> vec_kv_data = {&k_data, &v_data};
    for (const NDArray* data_ptr : vec_kv_data) {
      CHECK_EQ((*data_ptr)->ndim, 4) << error_msg;
      CHECK_EQ((*data_ptr)->shape[0], num_layers_)
          << error_msg << " The number of layers mismatches.";
      CHECK_EQ((*data_ptr)->shape[1], end_pos - start_pos)
          << error_msg << " The sequence length mismatches.";
      CHECK_EQ((*data_ptr)->shape[2], num_kv_heads_)
          << error_msg << " The number of heads mismatches.";
      CHECK_EQ((*data_ptr)->shape[3], qk_head_dim_)
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
    NDArray position_map_device = NDArray::Empty({end_pos - start_pos}, dtype_aux_, device_);
    position_map_device.CopyFromBytes(
        append_position_map.data() + start_pos,
        (end_pos - start_pos) * ((dtype_aux_.bits * dtype_aux_.lanes + 7) / 8));
    for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
      CHECK(attn_kinds_[layer_id] == AttnKind::kMHA) << "Only MHA is supported for DebugGetKV";
      f_debug_get_kv_.value()(pages_[layer_id], position_map_device, k_data, v_data, layer_id);
    }
  }

  void DebugGetKVMLA(int64_t seq_id, int64_t start_pos, int64_t end_pos, NDArray kv_data) final {
    CHECK(f_debug_get_kv_.defined())
        << "PageAttentionKVCache requires the `f_debug_get_kv` to be explicitly passed in when "
           "initialization. Please construct the KV cache with `f_debug_get_kv`.";

    const Sequence& seq = seq_map_.at(seq_id);
    CHECK_GE(start_pos, 0) << "DebugGetKV does not accept negative start_pos " << start_pos;
    CHECK_LE(end_pos, seq.seq_length) << "DebugGetKV does not accept out-of-range end_pos";
    CHECK_LT(start_pos, end_pos) << "DebugGetKV does not accept \"start_pos >= end_pos\"";

    // kv_data: (num_layers, seq_length, qk_head_dim)
    static constexpr const char* error_msg =
        "DebugGetKV expects the kv_data in layout (num_layers, seq_length, qk_head_dim).";
    CHECK_EQ(kv_data->ndim, 3) << error_msg;
    CHECK_EQ(kv_data->shape[0], num_layers_) << error_msg << " The number of layers mismatches.";
    CHECK_EQ(kv_data->shape[1], end_pos - start_pos)
        << error_msg << " The sequence length mismatches.";
    CHECK_EQ(kv_data->shape[2], qk_head_dim_)
        << error_msg << " The number of head features mismatches.";

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
    NDArray position_map_device = NDArray::Empty({end_pos - start_pos}, dtype_aux_, device_);
    position_map_device.CopyFromBytes(
        append_position_map.data() + start_pos,
        (end_pos - start_pos) * ((dtype_aux_.bits * dtype_aux_.lanes + 7) / 8));
    for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
      CHECK(attn_kinds_[layer_id] == AttnKind::kMLA) << "Only MHA is supported for DebugGetKVMLA";
      f_debug_get_kv_.value()(pages_[layer_id], position_map_device, kv_data, layer_id);
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

  void ConstructTokenTreeMask(const std::vector<Sequence*>& sequences,
                              const IntTuple& token_tree_parent_ptr,
                              const std::vector<std::vector<int32_t>>& block_ids_on_depths,
                              const std::vector<std::vector<int32_t>>& trailing_blocks) {
    // Check whether the token tree of a sequence should be handled at the current depth.
    auto check_for_sequence = [&](int seq_i, int depth) -> bool {
      if (!append_before_attn_) {
        return true;
      }
      // Check if the last block of the sequence is on the current depth.
      if (block_ids_on_depths[depth][seq_i] == sequences[seq_i]->last_block_idx ||
          (depth + 1 == kPagedKVCacheMaxBlockDepth && !trailing_blocks[seq_i].empty())) {
        return true;
      }
      return false;
    };
    for (int d = 0; d < num_depths_; ++d) {
      // We check if the token tree deteriorates to a chain,
      // because chain cases can have simplified attention work flow.
      ICHECK_LT(d, tree_attn_mask_host_.size());
      ICHECK_LT(d, tree_attn_mn_indptr_host_.size());
      HostMemoryVector& tree_attn_mn_indptr = tree_attn_mn_indptr_host_[d];
      HostMemoryVector& tree_attn_mask = tree_attn_mask_host_[d];

      std::vector<bool> seq_in_current_depth(cur_batch_size_, false);

      tree_attn_mn_indptr.clear();
      tree_attn_mask.clear();
      std::fill(is_chain_on_depths_.begin(), is_chain_on_depths_.end(), true);

      bool is_chain = true;
      // - Construct the mn indptr array, which is the indptr of the mask size of each sequence.
      tree_attn_mn_indptr.push_back(0);
      ICHECK_EQ(sequences.size(), cur_batch_size_);
      ICHECK_EQ(cur_append_lengths_.size(), cur_batch_size_);
      int64_t token_tree_parent_ptr_offset = 0;
      for (int i = 0; i < cur_batch_size_; ++i) {
        int64_t append_length = cur_append_lengths_[i];
        seq_in_current_depth[i] = check_for_sequence(i, d);
        if (!seq_in_current_depth[i]) {
          tree_attn_mn_indptr.push_back(tree_attn_mn_indptr.back());
          token_tree_parent_ptr_offset += append_length;  // Skip the token tree of this sequence.
          continue;
        }
        // Update the token tree parent pointers.
        CHECK_LE(sequences[i]->token_tree_parent_ptr.size(),
                 global_block_pool_[sequences[i]->last_block_idx].seq_length)
            << "The token tree size is larger than the sequence length of the last block.";
        std::copy(token_tree_parent_ptr.begin() + token_tree_parent_ptr_offset,
                  token_tree_parent_ptr.begin() + token_tree_parent_ptr_offset + append_length,
                  std::back_inserter(sequences[i]->token_tree_parent_ptr));
        token_tree_parent_ptr_offset += append_length;

        CHECK_LE(sequences[i]->token_tree_parent_ptr.size(), kTreeAttnMaxTreeSize)
            << "The tree size is " << append_length << " which exceeds the maximum tree size limit "
            << kTreeAttnMaxTreeSize;
        tree_attn_mn_indptr.push_back(tree_attn_mn_indptr.back() +
                                      sequences[i]->token_tree_parent_ptr.size());
      }
      CHECK_EQ(token_tree_parent_ptr.size(), token_tree_parent_ptr_offset)
          << "Invalid token tree size. The sum of \"append_lengths\" is "
          << token_tree_parent_ptr_offset << " while there are " << token_tree_parent_ptr.size()
          << " elements in \"token_tree_parent_ptr\".";

      // - Construct the mask of each sequence.
      for (int i = 0; i < cur_batch_size_; ++i) {
        if (!seq_in_current_depth[i]) {
          continue;
        }
        int64_t tree_size = sequences[i]->token_tree_parent_ptr.size();
        std::vector<std::vector<int32_t>> mask;
        std::vector<int32_t> depth;
        mask.reserve(tree_size);
        depth.reserve(tree_size);
        sequences[i]->is_chain = true;
        sequences[i]->accepted_indices_committed = false;
        std::unordered_map<int, std::vector<int>> tree_parent_to_children;
        std::vector<int> tree_roots;
        for (int n = 0; n < tree_size; ++n) {
          CHECK_LT(sequences[i]->token_tree_parent_ptr[n], n)
              << "Invalid token tree. The parent of node " << n << " in tree " << i << " is "
              << sequences[i]->token_tree_parent_ptr[n] << ", which is not smaller than " << n;
          CHECK_GE(sequences[i]->token_tree_parent_ptr[n], -1)
              << "Invalid token tree. The parent of node " << n << " in tree " << i << " is "
              << sequences[i]->token_tree_parent_ptr[n];
          if (sequences[i]->token_tree_parent_ptr[n] != n - 1) {
            // The parent of the current node is not the last node.
            // Therefore the tree is not a chain.
            sequences[i]->is_chain = false;
            is_chain = false;
          }
          tree_parent_to_children[sequences[i]->token_tree_parent_ptr[n]].push_back(n);

          if (sequences[i]->token_tree_parent_ptr[n] != -1) {
            depth.push_back(depth[sequences[i]->token_tree_parent_ptr[n]] + 1);
          } else {
            depth.push_back(0);
            tree_roots.push_back(n);
          }
        }
        std::vector<std::pair<int, int>> tree_order(tree_size);
        int order = 0;
        std::function<int(int)> tree_dfs = [&order, &tree_order, &tree_parent_to_children,
                                            &tree_dfs](int node) -> int {
          tree_order[node].first = order++;
          int upper_bound = tree_order[node].first + 1;
          for (int child : tree_parent_to_children[node]) {
            upper_bound = std::max(upper_bound, tree_dfs(child));
          }
          tree_order[node].second = upper_bound;
          return upper_bound;
        };
        for (auto root : tree_roots) {
          tree_dfs(root);
        }
        for (int n = 0; n < tree_size; ++n) {
          tree_attn_mask.push_back(tree_order[n].first);
          tree_attn_mask.push_back(tree_order[n].second);
        }
        sequences[i]->token_tree_node_depths = std::move(depth);
      }

      is_chain_on_depths_[d] = is_chain;

      if (!append_before_attn_) {
        break;
      }
    }
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
    CHECK_EQ(block.external_ref_cnt, 1)
        << "The block is " << block.external_ref_cnt - 1
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
   * on each depth. And if the depth is larger than the kPagedKVCacheMaxBlockDepth,
   * the exceeding blocks will concatenate and output separately.
   * More precisely, the inner returned vector contains the block ids
   * used by the sequences on a certain depth (or "-1" if a sequence
   * has fewer depth). The outer returned vector contains the inner
   * vectors from the lowest depth to the highest depth.
   */
  std::pair<std::vector<std::vector<int32_t>>, std::vector<std::vector<int32_t>>>
  GetBlockIdsOnDepth(const std::vector<Sequence*>& sequences) const {
    // - Get the trace of each sequence.
    int64_t num_depths = 0;
    std::vector<std::vector<int32_t>> seq_block_traces;
    std::vector<std::vector<int32_t>> trailing_block_traces;
    seq_block_traces.reserve(cur_batch_size_);
    trailing_block_traces.reserve(cur_batch_size_);
    for (int i = 0; i < cur_batch_size_; ++i) {
      std::vector<int32_t> trace = sequences[i]->GetBlockTrace(global_block_pool_);
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
      block_ids.reserve(cur_batch_size_);
      for (int i = 0; i < cur_batch_size_; ++i) {
        block_ids.push_back(
            d < static_cast<int>(seq_block_traces[i].size()) ? seq_block_traces[i][d] : -1);
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
  std::pair<std::vector<std::pair<int32_t, int32_t>>, bool> GetChunkedBlockIds(
      const std::vector<int32_t>& block_ids, bool enable_coalesce = true) const {
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
    bool use_decode_kernel = is_decode_request_ && coalesce_ratio < 32;
    return {use_decode_kernel || !enable_coalesce ? uncoalesced_block_ids : coalesced_block_ids,
            use_decode_kernel};
  }

  /*! \brief Check whether BeginForward for kernels is needed. */
  bool NeedKernelBeginForward() {
    return f_attention_prefill_begin_forward_.defined() &&
           f_attention_decode_begin_forward_.defined() &&
           f_attention_prefill_ragged_begin_forward_.defined();
  }

  /*! \brief Invoke the "begin forward" functions of underlying kernels. */
  void KernelBeginForward() {
    if (!NeedKernelBeginForward()) {
      return;
    }

    if (!append_before_attn_) {
      if (is_chain_on_depths_[0]) {
        f_attention_prefill_ragged_begin_forward_.value()(
            temp_float_attn_workspace_, temp_int_attn_workspace_[0],
            cur_append_lengths_indptr_host_.as_ndarray(),
            cur_append_lengths_indptr_host_.as_ndarray(), cur_batch_size_, num_qo_heads_,
            num_kv_heads_, qk_head_dim_, copy_stream_);
      }
    }
    for (int d = 0; d < num_depths_; ++d) {
      if (page_indices_on_depths_view_[d]->shape[0] == 0) {
        continue;
      }
      CHECK(!support_sliding_window_) << "Kernel BeginForward doesn't support sliding window.";
      if (use_decode_kernel_[d]) {
        f_attention_decode_begin_forward_.value()(
            d, temp_float_attn_workspace_, temp_int_attn_workspace_[d + 1],
            page_indptr_on_depths_host_[d].as_ndarray(),
            last_page_len_on_depths_host_[d].as_ndarray(), num_qo_heads_, num_kv_heads_,
            qk_head_dim_, page_size_,
            /*rotary_mode=*/rope_mode_ == RoPEMode::kInline, copy_stream_);
      } else {
        f_attention_prefill_begin_forward_.value()(
            /*depth=*/d, temp_float_attn_workspace_, temp_int_attn_workspace_[d + 1],
            qo_indptr_on_depths_host_[d].as_ndarray(), page_indptr_on_depths_host_[d].as_ndarray(),
            static_cast<int>(page_indptr_on_depths_host_[d].size()) - 1, num_qo_heads_,
            num_kv_heads_, qk_head_dim_, page_size_, copy_stream_);
      }
    }
  }

  /*!
   * \brief Compute attention for between the input q data and the
   * input k/v data and the k/v data in cache on the given layer.
   */
  void AttentionInternal(int64_t layer_id, NDArray q_data, NDArray k_data, NDArray v_data,
                         NDArray output, double attn_score_scaling_factor) {
    int64_t local_layer_id = layer_id - layer_id_begin_offset_;
    CHECK_GE(local_layer_id, 0);
    CHECK_LT(local_layer_id, num_layers_);
    PackedFunc f_prefill =
        !support_sliding_window_ ? f_attention_prefill_ : f_attention_prefill_sliding_window_;
    PackedFunc f_decode =
        !support_sliding_window_ ? f_attention_decode_ : f_attention_decode_sliding_window_;
    CHECK_GE(num_depths_, 1) << "The number of effective depths must be greater or equal to 1.";

    bool is_first_kernel = true;
    if (!append_before_attn_) {
      // The first part of attention, which only involves the q and the newly appended k/v.
      is_first_kernel = false;
      if (is_chain_on_depths_[0]) {
        // If the batch does not form a tree, use raggedness prefill kernel.
        f_attention_prefill_ragged_(q_data, cur_append_length_indptr_view_, k_data, v_data,
                                    cur_append_length_indptr_view_, q_rope_position_map_view_,
                                    k_ragged_rope_pos_offset_view_, output,
                                    merged_attn_scores_view_,
                                    /*causal=*/1,
                                    /*rotary_mode=*/rope_mode_ == RoPEMode::kInline, rotary_scale_,
                                    rotary_theta_, attn_score_scaling_factor);
      } else {
        // The batch requires tree attention.
        ICHECK(f_attention_prefill_with_tree_mask_.defined())
            << "Function \"f_attention_prefill_with_tree_mask_\" is not defined.";
        ICHECK(tree_attn_mask_view_[0].defined());
        ICHECK(tree_attn_mn_indptr_view_[0].defined());
        f_attention_prefill_with_tree_mask_(
            q_data, cur_append_length_indptr_view_, k_data, v_data, cur_append_length_indptr_view_,
            q_rope_position_map_view_, tree_attn_mn_indptr_view_[0], tree_attn_mask_view_[0],
            output, merged_attn_scores_view_, /*rotary_mode=*/rope_mode_ == RoPEMode::kInline,
            rotary_scale_, rotary_theta_, attn_score_scaling_factor, cur_batch_size_);
      }
    }

    for (int d = 0; d < num_depths_; ++d) {
      if (page_indices_on_depths_view_[d]->shape[0] == 0) {
        continue;
      }
      NDArray attn_output;
      NDArray attn_scores;
      if (is_first_kernel) {
        attn_output = output;
        attn_scores = merged_attn_scores_view_;
      } else {
        attn_output = temp_attn_output_view_;
        attn_scores = temp_attn_scores_view_;
      }
      if (append_before_attn_ && !is_chain_on_depths_[d]) {
        f_attention_prefill_with_tree_mask_paged_kv_(
            /*depth=*/d, q_data, qo_indptr_on_depths_view_[d], pages_[local_layer_id],
            page_indptr_on_depths_view_[d], page_indices_on_depths_view_[d],
            length_info_on_depths_view_[d], k_rope_pos_offset_view_[d], q_rope_position_map_view_,
            attn_output, attn_scores,
            /*rotary_mode=*/rope_mode_ == RoPEMode::kInline, rotary_scale_, rotary_theta_,
            attn_score_scaling_factor, tree_attn_mn_indptr_view_[d], tree_attn_mask_view_[d]);
      } else if (use_decode_kernel_[d]) {
        // Use decode kernel for depth d
        f_decode(/*depth=*/d, q_data, pages_[local_layer_id], page_indptr_on_depths_view_[d],
                 page_indices_on_depths_view_[d], length_info_on_depths_view_[d],
                 k_rope_pos_offset_view_[d], q_rope_position_map_view_, attn_output, attn_scores,
                 /*rotary_mode=*/rope_mode_ == RoPEMode::kInline, rotary_scale_, rotary_theta_,
                 attn_score_scaling_factor);
      } else {
        // Use prefill kernel for depth d
        f_prefill(/*depth=*/d, q_data, qo_indptr_on_depths_view_[d], pages_[local_layer_id],
                  page_indptr_on_depths_view_[d], page_indices_on_depths_view_[d],
                  length_info_on_depths_view_[d], k_rope_pos_offset_view_[d],
                  q_rope_position_map_view_, attn_output, attn_scores, /*causal=*/0,
                  /*rotary_mode=*/rope_mode_ == RoPEMode::kInline, rotary_scale_, rotary_theta_,
                  attn_score_scaling_factor);
      }

      if (!is_first_kernel) {
        f_merge_inplace_(output, merged_attn_scores_view_, temp_attn_output_view_,
                         temp_attn_scores_view_);
      } else {
        is_first_kernel = false;
      }
    }
  }

  void MLAAbsorbedInternal(int64_t layer_id, NDArray q_data, NDArray compressed_kv_data,
                           NDArray k_pe_data, NDArray output, double attn_score_scaling_factor) {
    int64_t local_layer_id = layer_id - layer_id_begin_offset_;
    CHECK_GE(local_layer_id, 0);
    CHECK_LT(local_layer_id, num_layers_);
    PackedFunc f_prefill = f_mla_prefill_;
    PackedFunc f_decode = f_mla_decode_;
    CHECK_GE(num_depths_, 1) << "The number of effective depths must be greater or equal to 1.";

    bool is_first_kernel = true;
    if (!append_before_attn_) {
      // The first part of attention, which only involves the q and the newly appended k/v.
      is_first_kernel = false;
      CHECK(is_chain_on_depths_[0]) << "Tree attn not able for MLA for now.";
      // If the batch does not form a tree, use raggedness prefill kernel.
      f_mla_prefill_ragged_absorbed_(q_data, cur_append_length_indptr_view_, compressed_kv_data,
                                     k_pe_data, cur_append_length_indptr_view_, output,
                                     merged_attn_scores_view_,
                                     /*causal=*/1, attn_score_scaling_factor);
    }

    for (int d = 0; d < num_depths_; ++d) {
      if (page_indices_on_depths_view_[d]->shape[0] == 0) {
        continue;
      }
      NDArray attn_output;
      NDArray attn_scores;
      if (is_first_kernel) {
        attn_output = output;
        attn_scores = merged_attn_scores_view_;
      } else {
        attn_output = temp_attn_output_view_;
        attn_scores = temp_attn_scores_view_;
      }
      CHECK(is_chain_on_depths_[d]) << "Tree attn not able for MLA for now.";
      if (use_decode_kernel_[d]) {
        // Use decode kernel for depth d
        f_decode(/*depth=*/d, q_data, pages_[local_layer_id], page_indptr_on_depths_view_[d],
                 page_indices_on_depths_view_[d], length_info_on_depths_view_[d], attn_output,
                 attn_scores, attn_score_scaling_factor);
      } else {
        // Use prefill kernel for depth d
        f_prefill(/*depth=*/d, q_data, qo_indptr_on_depths_view_[d], pages_[local_layer_id],
                  page_indptr_on_depths_view_[d], page_indices_on_depths_view_[d],
                  length_info_on_depths_view_[d], attn_output, attn_scores, /*causal=*/0,
                  attn_score_scaling_factor);
      }

      if (!is_first_kernel) {
        f_merge_inplace_(output, merged_attn_scores_view_, temp_attn_output_view_,
                         temp_attn_scores_view_);
      } else {
        is_first_kernel = false;
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
   * \brief Synchronize auxiliary arrays to device.
   * \note This method resets the dirty flag to false, and needs to be
   * invoked before running attention computation on device.
   */
  void SyncAuxArrayToDevice() {
    ICHECK(dtype_aux_.bits == 32 && dtype_aux_.code == kDLInt);
    int64_t total_append_length = 0;
    int num_sequences = cur_append_lengths_.size();
    cur_append_lengths_indptr_host_.clear();
    cur_append_lengths_indptr_host_.push_back(0);
    for (int i = 0; i < num_sequences; ++i) {
      cur_append_lengths_indptr_host_.push_back(cur_append_lengths_indptr_host_.back() +
                                                cur_append_lengths_[i]);
    }
    total_append_length = cur_append_lengths_indptr_host_.back();
    ICHECK_EQ(total_append_length, append_position_map_host_.size());
    ICHECK_EQ(total_append_length, kv_transfer_remote_position_map_host_.size());
    ICHECK_EQ(total_append_length, kv_transfer_recver_id_host_.size());

    // - Reset the copy.
    aux_data_manager_->ResetAttnAuxDataCopy();

    // 1. q_rope_position_map
    // q_rope_position_map has to be synced first so that it has a 0 byte offset
    ICHECK_EQ(q_rope_position_map_host_.size(), total_append_length);
    q_rope_position_map_view_ = aux_data_manager_->CopyQRoPEPosMapAsync(&q_rope_position_map_host_);
    // 2. qo_indptr_on_depths
    for (int d = 0; d < num_depths_; ++d) {
      qo_indptr_on_depths_view_[d] =
          aux_data_manager_->CopyQOIndptrOnDepthAsync(&qo_indptr_on_depths_host_[d], d);
    }
    // 3. page_indptr_on_depths
    for (int d = 0; d < num_depths_; ++d) {
      ICHECK_EQ(page_indptr_on_depths_host_[d].size(), qo_indptr_on_depths_host_[d].size());
      page_indptr_on_depths_view_[d] =
          aux_data_manager_->CopyPageIndptrOnDepthAsync(&page_indptr_on_depths_host_[d], d);
    }
    // 4. page_indices_on_depths
    for (int d = 0; d < num_depths_; ++d) {
      ICHECK_EQ(page_indices_on_depths_host_[d].size(), page_indptr_on_depths_host_[d].back());
      page_indices_on_depths_view_[d] =
          aux_data_manager_->CopyPageIndicesOnDepthAsync(&page_indices_on_depths_host_[d], d);
    }
    // 5. length_info_on_depths
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
            aux_data_manager_->CopyLastPageLenOnDepthAsync(&last_page_len_on_depths_host_[d], d);
      } else {
        // Sliding window is enabled,
        length_info_on_depths_view_[d] = aux_data_manager_->CopyLengthInfoOnDepthAsync(
            &last_page_len_on_depths_host_[d], &sliding_window_offset_on_depths_host_[d],
            &sink_size_on_depths_host_[d], d);
      }
    }
    // 6. k_rope_pos_offset_on_depths
    for (int d = 0; d < num_depths_; ++d) {
      ICHECK_EQ(k_rope_pos_offset_on_depths_host_[d].size() + 1,
                qo_indptr_on_depths_host_[d].size());
      k_rope_pos_offset_view_[d] = aux_data_manager_->CopyKRoPEPosOffsetOnDepthAsync(
          &k_rope_pos_offset_on_depths_host_[d], d);
    }
    // 7. cur_append_lengths_indptr
    cur_append_length_indptr_view_ =
        aux_data_manager_->CopyCurAppendLengthIndptrAsync(&cur_append_lengths_indptr_host_);
    // 8. k_ragged_rope_pos_offset
    ICHECK_EQ(k_ragged_rope_pos_offset_host_.size(), num_sequences);
    k_ragged_rope_pos_offset_view_ =
        aux_data_manager_->CopyKRaggedRoPEPosOffsetAsync(&k_ragged_rope_pos_offset_host_);
    // 9. append_position_map
    append_position_map_view_ =
        aux_data_manager_->CopyAppendPositionMapAsync(&append_position_map_host_);
    // 10. kv_transfer_remote_position_map
    kv_transfer_remote_position_map_view_ = aux_data_manager_->CopyKVTransferRemotePositionMapAsync(
        &kv_transfer_remote_position_map_host_);
    // 11. kv_transfer_recver_id
    kv_transfer_recver_id_view_ =
        aux_data_manager_->CopyKVTransferRecverIDAsync(&kv_transfer_recver_id_host_);

    // 12. kv_transfer_page_to_page_local_position_map
    kv_transfer_page_to_page_local_position_map_view_ =
        aux_data_manager_->CopyKVTransferPage2PageLocalPositionMapAsync(
            &kv_transfer_page_to_page_local_position_map_host_);
    // 13. kv_transfer_page_to_page_remote_position_map
    kv_transfer_page_to_page_remote_position_map_view_ =
        aux_data_manager_->CopyKVTransferPage2PageRemotePositionMapAsync(
            &kv_transfer_page_to_page_remote_position_map_host_);
    // 14. kv_transfer_page_to_page_recver_id
    kv_transfer_page_to_page_recver_id_view_ =
        aux_data_manager_->CopyKVTransferPage2PageRecverIDAsync(
            &kv_transfer_page_to_page_recver_id_host_);
    // 15. tree_attn_mask and tree_attn_mn_indptr
    for (int d = 0; d < num_depths_; ++d) {
      if (!is_chain_on_depths_[d]) {
        tree_attn_mask_view_[d] =
            aux_data_manager_->CopyTreeAttnMaskOnDepthAsync(&tree_attn_mask_host_[d], d);
        tree_attn_mn_indptr_view_[d] =
            aux_data_manager_->CopyTreeAttnMNIndptrOnDepthAsync(&tree_attn_mn_indptr_host_[d], d);
      }
    }
    // 16. Create view for temporary arrays for attention computation.
    temp_attn_output_view_ = temp_attn_output_device_.CreateView(
        {total_append_length, num_qo_heads_, v_head_dim_}, temp_attn_output_device_->dtype);
    temp_attn_scores_view_ = temp_attn_scores_device_.CreateView(
        {total_append_length, num_qo_heads_}, temp_attn_scores_device_->dtype);
    merged_attn_scores_view_ = merged_attn_scores_device_.CreateView(
        {total_append_length, num_qo_heads_}, merged_attn_scores_device_->dtype);

    // - Commit the copy.
    aux_data_manager_->CommitAttnAuxDataCopy();
    // - Reset the dirty flag to false.
    dirty_aux_data_device_ = false;
  }
};  // namespace relax_vm

TVM_REGISTER_OBJECT_TYPE(PagedAttentionKVCacheObj);

//-------------------------------------------------
//  Register runtime functions
//-------------------------------------------------

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_create")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.size() == 29 || args.size() == 30)
          << "Invalid number of KV cache constructor args.";
      ShapeTuple cache_config = args[0];
      ShapeTuple layer_indptr_tuple = args[1];
      int num_groups = 1;
      int group_id = 0;
      if (DiscoWorker* disco_worker = ThreadLocalDiscoWorker::Get()->worker) {
        // In the Disco worker thread
        num_groups = disco_worker->num_groups;
        group_id = disco_worker->worker_id / (disco_worker->num_workers / num_groups);
      }
      CHECK_EQ(layer_indptr_tuple.size(), num_groups + 1);
      int64_t num_layers = layer_indptr_tuple[group_id + 1] - layer_indptr_tuple[group_id];
      int64_t layer_id_begin_offset = layer_indptr_tuple[group_id];
      int64_t num_qo_heads = args[2];
      int64_t num_kv_heads = args[3];
      int64_t head_dim = args[4];
      int rope_mode = args[5];
      double rotary_scale = args[6];
      double rotary_theta = args[7];
      NDArray init = args[8];
      PackedFunc f_transpose_append = args[9];
      PackedFunc f_attention_prefill = args[10];
      PackedFunc f_attention_decode = args[11];
      PackedFunc f_attention_prefill_sliding_window = args[12];
      PackedFunc f_attention_decode_sliding_window = args[13];
      PackedFunc f_attention_prefill_ragged = args[14];
      PackedFunc f_attention_prefill_ragged_begin_forward = args[15];
      PackedFunc f_attention_prefill_ragged_end_forward = args[16];
      PackedFunc f_attention_prefill_begin_forward = args[17];
      PackedFunc f_attention_prefill_end_forward = args[18];
      PackedFunc f_attention_decode_begin_forward = args[19];
      PackedFunc f_attention_decode_end_forward = args[20];
      PackedFunc f_merge_inplace = args[21];
      PackedFunc f_split_rotary = args[22];
      PackedFunc f_copy_single_page = args[23];
      Optional<PackedFunc> f_debug_get_kv = args[24];
      PackedFunc f_compact_copy = args[25];
      PackedFunc f_attention_prefill_with_tree_mask = args[26];
      PackedFunc f_attention_prefill_with_tree_mask_paged_kv = args[27];
      Optional<NDArray> rope_ext_factors = NullOpt;
      bool enable_kv_transfer = false;

      if (args[28].IsObjectRef<NDArray>()) {
        rope_ext_factors = args[28].AsObjectRef<NDArray>();
      }
      if (args.size() >= 30) {
        enable_kv_transfer = args[29];
      }

      std::vector<AttnKind> attn_kinds(/*size=*/layer_indptr_tuple[num_groups],
                                       /*value=*/AttnKind::kMHA);

      CHECK_EQ(cache_config.size(), 5);
      int64_t reserved_num_seqs = cache_config[0];
      int64_t total_token_capacity = cache_config[1];
      int64_t prefill_chunk_size = cache_config[2];
      int64_t page_size = cache_config[3];
      bool support_sliding_window = cache_config[4];
      int64_t num_total_pages = (total_token_capacity + page_size - 1) / page_size + 1;
      if (support_sliding_window) {
        // When sliding window is enabled, each sequence may use two more pages at most.
        num_total_pages += reserved_num_seqs * 2;
      }
      // NOTE: We will remove this legacy construction after finishing the transition phase.
      // Some `PackedFunc()` here are placeholders that will be filled.
      ObjectPtr<PagedAttentionKVCacheObj> n = make_object<PagedAttentionKVCacheObj>(
          page_size, num_layers, layer_id_begin_offset, num_qo_heads, num_kv_heads, head_dim,
          head_dim, /*qk_rope_head_dim=*/0, attn_kinds, reserved_num_seqs, num_total_pages,
          prefill_chunk_size, support_sliding_window, RoPEMode(rope_mode), rotary_scale,
          rotary_theta,
          std::move(rope_ext_factors),                    //
          enable_kv_transfer, init->dtype, init->device,  //
          std::move(f_transpose_append), PackedFunc(), std::move(f_compact_copy),
          std::move(f_attention_prefill), std::move(f_attention_decode),
          std::move(f_attention_prefill_sliding_window),
          std::move(f_attention_decode_sliding_window), std::move(f_attention_prefill_ragged),
          std::move(f_attention_prefill_with_tree_mask),
          std::move(f_attention_prefill_with_tree_mask_paged_kv),
          std::move(f_attention_prefill_ragged_begin_forward),
          std::move(f_attention_prefill_ragged_end_forward),
          std::move(f_attention_prefill_begin_forward), std::move(f_attention_prefill_end_forward),
          std::move(f_attention_decode_begin_forward), std::move(f_attention_decode_end_forward),
          PackedFunc(), PackedFunc(), PackedFunc(), PackedFunc(), std::move(f_merge_inplace),
          std::move(f_split_rotary), std::move(f_copy_single_page), std::move(f_debug_get_kv));
      *rv = AttentionKVCache(std::move(n));
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_create_reduced")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.size() == 23 || args.size() == 24)
          << "Invalid number of KV cache constructor args.";
      ShapeTuple cache_config = args[0];
      ShapeTuple layer_indptr_tuple = args[1];
      int num_groups = 1;
      int group_id = 0;
      if (DiscoWorker* disco_worker = ThreadLocalDiscoWorker::Get()->worker) {
        // In the Disco worker thread
        num_groups = disco_worker->num_groups;
        group_id = disco_worker->worker_id / (disco_worker->num_workers / num_groups);
      }
      CHECK_EQ(layer_indptr_tuple.size(), num_groups + 1);
      int64_t num_layers = layer_indptr_tuple[group_id + 1] - layer_indptr_tuple[group_id];
      int64_t layer_id_begin_offset = layer_indptr_tuple[group_id];
      int64_t num_qo_heads = args[2];
      int64_t num_kv_heads = args[3];
      int64_t head_dim = args[4];
      int rope_mode = args[5];
      double rotary_scale = args[6];
      double rotary_theta = args[7];
      NDArray init = args[8];
      PackedFunc f_transpose_append = args[9];
      PackedFunc f_attention_prefill = args[10];
      PackedFunc f_attention_decode = args[11];
      PackedFunc f_attention_prefill_sliding_window = args[12];
      PackedFunc f_attention_decode_sliding_window = args[13];
      PackedFunc f_attention_prefill_ragged = args[14];
      PackedFunc f_merge_inplace = args[15];
      PackedFunc f_split_rotary = args[16];
      PackedFunc f_copy_single_page = args[17];
      Optional<PackedFunc> f_debug_get_kv = args[18];
      PackedFunc f_compact_copy = args[19];
      PackedFunc f_attention_prefill_with_tree_mask = args[20];
      PackedFunc f_attention_prefill_with_tree_mask_paged_kv = args[21];
      Optional<NDArray> rope_ext_factors = NullOpt;
      bool enable_kv_transfer = false;

      if (args[22].IsObjectRef<NDArray>()) {
        rope_ext_factors = args[22].AsObjectRef<NDArray>();
      }
      if (args.size() >= 24) {
        enable_kv_transfer = args[23];
      }

      std::vector<AttnKind> attn_kinds(/*size=*/layer_indptr_tuple[num_groups],
                                       /*value=*/AttnKind::kMHA);

      CHECK_EQ(cache_config.size(), 5);
      int64_t reserved_num_seqs = cache_config[0];
      int64_t total_token_capacity = cache_config[1];
      int64_t prefill_chunk_size = cache_config[2];
      int64_t page_size = cache_config[3];
      bool support_sliding_window = cache_config[4];
      int64_t num_total_pages = (total_token_capacity + page_size - 1) / page_size + 1;
      if (support_sliding_window) {
        // When sliding window is enabled, each sequence may use two more pages at most.
        num_total_pages += reserved_num_seqs * 2;
      }
      // NOTE: We will remove this legacy construction after finishing the transition phase.
      // Some `PackedFunc()` here are placeholders that will be filled.
      ObjectPtr<PagedAttentionKVCacheObj> n = make_object<PagedAttentionKVCacheObj>(
          page_size, num_layers, layer_id_begin_offset, num_qo_heads, num_kv_heads, head_dim,
          head_dim, /*qk_rope_head_dim=*/0, attn_kinds, reserved_num_seqs, num_total_pages,
          prefill_chunk_size, support_sliding_window, RoPEMode(rope_mode), rotary_scale,
          rotary_theta,
          std::move(rope_ext_factors),                    //
          enable_kv_transfer, init->dtype, init->device,  //
          std::move(f_transpose_append), PackedFunc(), std::move(f_compact_copy),
          std::move(f_attention_prefill), std::move(f_attention_decode),
          std::move(f_attention_prefill_sliding_window),
          std::move(f_attention_decode_sliding_window), std::move(f_attention_prefill_ragged),
          std::move(f_attention_prefill_with_tree_mask),           //
          std::move(f_attention_prefill_with_tree_mask_paged_kv),  //
          NullOpt, NullOpt, NullOpt, NullOpt, NullOpt, NullOpt,    //
          PackedFunc(), PackedFunc(), PackedFunc(), PackedFunc(), std::move(f_merge_inplace),
          std::move(f_split_rotary), std::move(f_copy_single_page), std::move(f_debug_get_kv));
      *rv = AttentionKVCache(std::move(n));
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_create_reduced_mla")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.size() == 38) << "Invalid number of KV cache constructor args.";
      ShapeTuple cache_config = args[0];
      ShapeTuple layer_indptr_tuple = args[1];
      int num_groups = 1;
      int group_id = 0;
      if (DiscoWorker* disco_worker = ThreadLocalDiscoWorker::Get()->worker) {
        // In the Disco worker thread
        num_groups = disco_worker->num_groups;
        group_id = disco_worker->worker_id / (disco_worker->num_workers / num_groups);
      }
      CHECK_EQ(layer_indptr_tuple.size(), num_groups + 1);
      int64_t num_layers = layer_indptr_tuple[group_id + 1] - layer_indptr_tuple[group_id];
      int64_t layer_id_begin_offset = layer_indptr_tuple[group_id];
      int64_t num_qo_heads = args[2];
      int64_t num_kv_heads = args[3];
      int64_t qk_head_dim = args[4];
      int64_t v_head_dim = args[5];
      int64_t qk_rope_head_dim = args[6];
      IntTuple attn_kinds = args[7];
      int rope_mode = args[8];
      double rotary_scale = args[9];
      double rotary_theta = args[10];
      NDArray init = args[11];
      PackedFunc f_transpose_append = args[12];
      PackedFunc f_transpose_append_mla = args[13];
      PackedFunc f_attention_prefill = args[14];
      PackedFunc f_attention_decode = args[15];
      PackedFunc f_attention_prefill_sliding_window = args[16];
      PackedFunc f_attention_decode_sliding_window = args[17];
      PackedFunc f_attention_prefill_ragged = args[18];
      Optional<PackedFunc> f_attention_prefill_ragged_begin_forward = NullOpt;
      Optional<PackedFunc> f_attention_prefill_ragged_end_forward = NullOpt;
      Optional<PackedFunc> f_attention_prefill_begin_forward = NullOpt;
      Optional<PackedFunc> f_attention_prefill_end_forward = NullOpt;
      Optional<PackedFunc> f_attention_decode_begin_forward = NullOpt;
      Optional<PackedFunc> f_attention_decode_end_forward = NullOpt;
      PackedFunc f_mla_prefill = args[25];
      PackedFunc f_mla_decode = args[26];
      PackedFunc f_mla_prefill_ragged_normal = args[27];
      PackedFunc f_mla_prefill_ragged_absorbed = args[28];
      PackedFunc f_merge_inplace = args[29];
      PackedFunc f_split_rotary = args[30];
      PackedFunc f_copy_single_page = args[31];
      Optional<PackedFunc> f_debug_get_kv = args[32];
      PackedFunc f_compact_copy = args[33];
      PackedFunc f_attention_prefill_with_tree_mask = args[34];
      PackedFunc f_attention_prefill_with_tree_mask_paged_kv = args[35];
      Optional<NDArray> rope_ext_factors = NullOpt;
      bool enable_kv_transfer = false;

      if (args[36].IsObjectRef<NDArray>()) {
        rope_ext_factors = args[36].AsObjectRef<NDArray>();
      }
      enable_kv_transfer = args[37];

      auto f_convert_optional_packed_func = [&args](int arg_idx) -> Optional<PackedFunc> {
        if (args[arg_idx].IsObjectRef<PackedFunc>()) {
          return args[arg_idx].AsObjectRef<PackedFunc>();
        }
        return NullOpt;
      };
      f_attention_prefill_ragged_begin_forward = f_convert_optional_packed_func(19);
      f_attention_prefill_ragged_end_forward = f_convert_optional_packed_func(20);
      f_attention_prefill_begin_forward = f_convert_optional_packed_func(21);
      f_attention_prefill_end_forward = f_convert_optional_packed_func(22);
      f_attention_decode_begin_forward = f_convert_optional_packed_func(23);
      f_attention_decode_end_forward = f_convert_optional_packed_func(24);

      std::vector<AttnKind> attn_kinds_vec;
      attn_kinds_vec.reserve(attn_kinds.size());
      for (int64_t attn_kind : attn_kinds) {
        attn_kinds_vec.push_back(static_cast<AttnKind>(attn_kind));
      }

      CHECK_EQ(cache_config.size(), 5);
      int64_t reserved_num_seqs = cache_config[0];
      int64_t total_token_capacity = cache_config[1];
      int64_t prefill_chunk_size = cache_config[2];
      int64_t page_size = cache_config[3];
      bool support_sliding_window = cache_config[4];
      int64_t num_total_pages = (total_token_capacity + page_size - 1) / page_size + 1;
      if (support_sliding_window) {
        // When sliding window is enabled, each sequence may use two more pages at most.
        num_total_pages += reserved_num_seqs * 2;
      }
      // NOTE: We will remove this legacy construction after finishing the transition phase.
      // Some `PackedFunc()` here are placeholders that will be filled.
      ObjectPtr<PagedAttentionKVCacheObj> n = make_object<PagedAttentionKVCacheObj>(
          page_size, num_layers, layer_id_begin_offset, num_qo_heads, num_kv_heads, qk_head_dim,
          v_head_dim, qk_rope_head_dim, attn_kinds_vec, reserved_num_seqs, num_total_pages,
          prefill_chunk_size, support_sliding_window, RoPEMode(rope_mode), rotary_scale,
          rotary_theta,
          std::move(rope_ext_factors),                    //
          enable_kv_transfer, init->dtype, init->device,  //
          std::move(f_transpose_append), std::move(f_transpose_append_mla),
          std::move(f_compact_copy), std::move(f_attention_prefill), std::move(f_attention_decode),
          std::move(f_attention_prefill_sliding_window),
          std::move(f_attention_decode_sliding_window), std::move(f_attention_prefill_ragged),
          std::move(f_attention_prefill_with_tree_mask),           //
          std::move(f_attention_prefill_with_tree_mask_paged_kv),  //
          std::move(f_attention_prefill_ragged_begin_forward),
          std::move(f_attention_prefill_ragged_end_forward),
          std::move(f_attention_prefill_begin_forward), std::move(f_attention_prefill_end_forward),
          std::move(f_attention_decode_begin_forward), std::move(f_attention_decode_end_forward),
          std::move(f_mla_prefill), std::move(f_mla_decode), std::move(f_mla_prefill_ragged_normal),
          std::move(f_mla_prefill_ragged_absorbed), std::move(f_merge_inplace),
          std::move(f_split_rotary), std::move(f_copy_single_page), std::move(f_debug_get_kv));
      *rv = AttentionKVCache(std::move(n));
    });

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
