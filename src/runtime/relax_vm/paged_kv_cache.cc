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
 * \brief The paged KV cache for attention.
 * - It supports managing the K/V data of **multiple sequences**.
 * - It manages K/V values by doing paging along the sequence-length
 * dimension with a configured page size.
 * - To add a sequence to the cache, use AddSequence.
 * - The basic example use of the paged KV cache after initialization
 * in each round of model forwarding is the following:
 *   - step 1. use `ResetAppendLengths` to reset the appending information
 *     for preparation,
 *   - step 2. use `ReserveExtraLengthForAppend` to specify the length
 *     of K/V data to be appended for each sequence,
 *   - step 3. use `SyncAuxArrayToDevice` to synchronize auxiliary arrays
 *     to device for append/attention computation,
 *   - step 4. for each layer, use `Append` to append the K/V data to the
 *     cache, and then use `Attention` to compute attention results with
 *     Q data.
 */
class PagedAttentionKVCacheObj : public Object {
 private:
  /*! \brief The total number of sequences managed in the KV cache. */
  int64_t num_total_seqs_ = 0;
  /*! \brief The number of pages that are in use by the sequences. */
  int64_t num_pages_in_use_ = 0;
  /*!
   * \brief The number of allocated pages, including the in-use pages
   * and the pages released due to sequence removal.
   */
  int64_t num_pages_allocated_ = 0;

  /********************* Configuration *********************/

  /*! \brief The page size (the sequence length each page manages) of the cache. */
  const int64_t page_size_;
  /*! \brief The number of layers in the model. */
  const int64_t num_layers_;
  /*! \brief The number of heads in the model. */
  const int64_t num_heads_;
  /*! \brief The number of features each head has. */
  const int64_t head_dim_;
  /*! \brief A boolean denoting if cache automatic growth is allowed. */
  const bool allow_growth_;

  /*! \brief We fix int32 to be the index dtype of auxiliary data. */
  const DLDataType dtype_aux_ = DLDataType(DataType::Int(32, 1));

  /********************* Page Structures *********************/

  /*!
   * \brief The KV data managed by the KV cache.
   * It has layout (num_pages, num_layers, 2, num_heads, page_size, head_dim).
   * Along on the "2" dimension, index 0 stands for K and 1 stands for V.
   */
  NDArray pages_;
  /*! \brief The list of ids of released pages for page reuse. */
  std::vector<int32_t> free_page_ids_;

  /*! \brief The list of page ids assigned for each sequence in the cache. */
  std::vector<std::vector<int32_t>> page_table_;
  /*! \brief The lengths of each sequence in the cache. */
  std::vector<int32_t> seq_lengths_;

  /********************* Current Batch Info *********************/

  /*!
   * \brief The current lengths to append for each sequence.
   * - The new K/V data appended to the cache must have the same length
   * as stored in this array.
   * - The Q data passed in for attention must also have the same length
   * as stored.
   * \note Invoke "ResetAppendLengths" to reset this array to all-zero.
   */
  std::vector<int64_t> cur_append_lengths_;

  /********************* Auxiliary Arrays on Device *********************/
  //-------------------------------------------
  // The following fields are auxiliary arrays on device.
  // All of them are directly derivable from the fields above.
  // We store them for efficient execution of attentions,
  // cache append, etc.
  //-------------------------------------------
  /*!
   * \brief A boolean flag indicating if the auxiliary arrays are dirty.
   * If it is dirty, an explicit "SyncAuxArrayToDevice" should be invoked.
   */
  bool dirty_aux_data_device_ = false;
  /*!
   * \brief The page table indptr array on device.
   * \note Since page table is a ragged data structure, we represent it
   * in CSR format (which uses indptr and values below) on device.
   */
  NDArray page_table_indptr_device_;
  /*! \brief The page table value array on device. */
  NDArray page_table_values_device_;
  /*!
   * \brief The array storing "the number of used slots in the last page
   * of each sequence" on device. Its values range in (0, page_size_].
   */
  NDArray last_page_offset_device_;
  /*!
   * \brief The append_length indptr array on device.
   * \note Since the Q/K/V data may have raggedness in terms of lengths,
   * we represent the the append lengths in CSR format.
   */
  NDArray cur_append_length_indptr_device_;
  /*!
   * \brief The corresponding sequence id for each position along the
   * length dimension of K/V data. It is used for efficient computation.
   */
  NDArray cur_pos2seqid_device_;

  //-------------------------------------------
  // For efficient memory management, the actual sizes of the arrays
  // above are over allocated.
  // We create a view for the actual shapes of each of the arrays
  // after each synchronization and pass these views as input for
  // attention/append.
  //-------------------------------------------
  NDArray page_table_indptr_view_;
  NDArray page_table_values_view_;
  NDArray last_page_offset_view_;
  NDArray cur_append_length_indptr_view_;
  NDArray cur_pos2seqid_view_;

 public:
  /*! \brief Constructor. Take the cache configuration and initialize the NDArrays. */
  explicit PagedAttentionKVCacheObj(int64_t page_size, int64_t num_layers, int64_t num_heads,
                                    int64_t head_dim, int64_t reserved_num_seqs,
                                    int64_t reserved_num_pages, DLDataType dtype, DLDevice device,
                                    bool allow_growth)
      : page_size_(page_size),
        num_layers_(num_layers),
        num_heads_(num_heads),
        head_dim_(head_dim),
        allow_growth_(allow_growth) {
    pages_ = NDArray::Empty({reserved_num_pages, num_layers, 2, num_heads, page_size, head_dim},
                            dtype, device);
    page_table_indptr_device_ = NDArray::Empty({reserved_num_seqs + 1}, dtype_aux_, device);
    page_table_values_device_ = NDArray::Empty({reserved_num_pages}, dtype_aux_, device);
    last_page_offset_device_ = NDArray::Empty({reserved_num_seqs}, dtype_aux_, device);
    cur_append_length_indptr_device_ = NDArray::Empty({reserved_num_seqs + 1}, dtype_aux_, device);
    cur_pos2seqid_device_ = NDArray::Empty({reserved_num_pages * page_size}, dtype_aux_, device);
  }

  /*!
   * \brief Add a sequence to the KV cache. Returns the sequence id.
   * \note This method adds a new sequence with **initial length zero**.
   * Call `ReserveExtraLengthForAppend` and `Append` afterwards to reserve
   * the append length and append data to the cache.
   * \returns The id of the new sequence.
   */
  int64_t AddSequence() {
    page_table_.push_back({});
    seq_lengths_.push_back(0);
    cur_append_lengths_.push_back(0);
    int64_t seq_id = num_total_seqs_++;
    return seq_id;
  }

  /*!
   * \brief Given a sequence id and a required extra length, allocate new
   * new pages for the sequence until the total capacity can cover the
   * current sequence length plus the required extra length.
   * \note By reserving extra length for the sequence, the subsequent appended
   * K/V data for the sequence must have the same length as the input length here.
   * \param seq_id The id of the sequence to process.
   * \param extra_length The extra length to reserve for the sequence.
   */
  void ReserveExtraLengthForAppend(int64_t seq_id, int64_t extra_length) {
    CHECK_GE(seq_id, 0) << "Input sequence id should be positive";
    CHECK_LT(seq_id, num_total_seqs_)
        << "Invalid input sequence id " << seq_id << ", which is out of the range of [0, "
        << num_total_seqs_ << ").";
    CHECK_GT(extra_length, 0) << "The input length should be positive.";

    // The reservation is based on the current sequence length.
    // If "current sequence + input extra length" does not exceed the
    // current capacity (number of pages * page size), no action is taken.
    int64_t cur_npage = page_table_[seq_id].size();
    int64_t tgt_npage = (seq_lengths_[seq_id] + extra_length + page_size_ - 1) / page_size_;
    for (int64_t page_idx = cur_npage; page_idx < tgt_npage; ++page_idx) {
      AllocatePageForSequence(seq_id);
    }
    seq_lengths_[seq_id] += extra_length;
    cur_append_lengths_[seq_id] += extra_length;
    dirty_aux_data_device_ = true;
  }

  /*!
   * \brief The entrance of attention. It takes an attention compute
   * function together with the query data, invokes the attention
   * function to complete the computation.
   * \param f_attention The input attention compute function.
   * \param q_data The query data. We support the following two layout settings:
   * - in batch decode settings, q_data has layout (num_total_seqs, 1, num_heads, head_dim)
   *   where num_total_seqs should exactly equal to the number of sequences in the cache.
   * - in other settings (single-sequence prefill, batch prefill, or speculation
   *   verification), q_data has the **flattened layout** to handle raggedness:
   *   (1, total query length, num_heads, head_dim).
   * \param layer_id The model layer index of the current attention.
   * \param output The attention output array.
   * \param apply_rotary A boolean flag indicating if to apply RoPE to Q and K
   * in attention computation.
   * \param rotary_scale The RoPE scale if applicable.
   * \param rotary_theta The RoPE theta if applicable.
   * \note As a TODO item, we could move the rope related parameters to the
   * f_attention closure in the future.
   */
  void Attention(PackedFunc f_attention, NDArray q_data, int64_t layer_id, NDArray output,
                 bool apply_rotary = false, double rotary_scale = 1.0f, double rotary_theta = 1e4) {
    // Check q_data shape validity.
    CHECK_EQ(q_data->ndim, 4);
    CHECK_GT(q_data->shape[1], 0);
    CHECK_EQ(q_data->shape[2], num_heads_);
    CHECK_EQ(q_data->shape[3], head_dim_);
    CHECK(q_data.DataType() == pages_.DataType());

    if (q_data->shape[0] > 1) {
      CHECK_EQ(q_data->shape[0], num_total_seqs_);
      CHECK_EQ(q_data->shape[1], 1);
    }
    int64_t ntoken = 0;
    for (int64_t seq_id = 0; seq_id < num_total_seqs_; ++seq_id) {
      ntoken += cur_append_lengths_[seq_id];
      CHECK_LE(cur_append_lengths_[seq_id], seq_lengths_[seq_id]);
      if (q_data->shape[0] > 1) {
        CHECK_EQ(cur_append_lengths_[seq_id], 1);
      }
    }
    CHECK_EQ(ntoken, q_data->shape[0] * q_data->shape[1]);

    // The auxiliary data structure on device must have been synchronized.
    CHECK(!dirty_aux_data_device_)
        << "The auxiliary arrays are not synchronized to device. Please call "
           "`SyncAuxArrayToDevice` to synchronize before calling `Attention`.";

    f_attention(q_data, pages_,                                          //
                page_table_indptr_view_, page_table_values_view_,        //
                last_page_offset_view_, cur_append_length_indptr_view_,  //
                layer_id, output, apply_rotary, rotary_scale, rotary_theta);
  }

  /*!
   * \brief Append the k/v data to the cache on the given layer.
   * Prior to Append, ReserveExtraLengthForAppend should be called to specify
   * the length of append for each request.
   * \param f_transpose_append The function that copies the input data to the
   * cache data. It does a transpose-copy due to the layout difference between
   * K/V and the cache data.
   * \param k_data The input k data.
   * \param v_data The input v data.
   * \param layer_id The model layer index of the current append.
   * \note We support the following two layout settings for K/V data:
   * - in batch decode settings, k/v_data has layout (num_total_seqs, 1, num_heads, head_dim)
   *   where num_total_seqs should exactly equal to the number of sequences in the cache.
   * - in other settings (single-sequence prefill, batch prefill, or speculation
   *   verification), k/v_data has the **flattened layout** to handle raggedness:
   *   (1, total query length, num_heads, head_dim).
   */
  void Append(PackedFunc f_transpose_append, NDArray k_data, NDArray v_data, int64_t layer_id) {
    // Check k/v_data shape validity
    CHECK_EQ(k_data->ndim, 4);
    CHECK_GT(k_data->shape[1], 0);
    CHECK_EQ(k_data->shape[2], num_heads_);
    CHECK_EQ(k_data->shape[3], head_dim_);
    for (int i = 0; i < 4; ++i) {
      CHECK_EQ(k_data->shape[i], v_data->shape[i]);
    }
    CHECK(k_data.DataType() == pages_.DataType());
    CHECK(v_data.DataType() == pages_.DataType());

    if (k_data->shape[0] > 1) {
      CHECK_EQ(k_data->shape[0], num_total_seqs_);
      CHECK_EQ(k_data->shape[1], 1);
    }
    int64_t ntoken = 0;
    for (int64_t seq_id = 0; seq_id < num_total_seqs_; ++seq_id) {
      ntoken += cur_append_lengths_[seq_id];
      CHECK_LE(cur_append_lengths_[seq_id], seq_lengths_[seq_id]);
      if (k_data->shape[0] > 1) {
        CHECK_EQ(cur_append_lengths_[seq_id], 1);
      }
    }
    CHECK_EQ(ntoken, k_data->shape[0] * k_data->shape[1]);
    ICHECK_EQ(ntoken, cur_pos2seqid_view_->shape[0]);

    // The auxiliary data structure on device must have been synchronized.
    CHECK(!dirty_aux_data_device_)
        << "The auxiliary arrays are not synchronized to device. Please call "
           "`SyncAuxArrayToDevice` to synchronize before calling `Append`.";

    // Copy data
    f_transpose_append(pages_,  //
                       k_data.CreateView({ntoken, num_heads_, head_dim_}, k_data->dtype),
                       v_data.CreateView({ntoken, num_heads_, head_dim_}, v_data->dtype),
                       page_table_indptr_view_, page_table_values_view_,  //
                       last_page_offset_view_, cur_append_length_indptr_view_, cur_pos2seqid_view_,
                       layer_id);
  }

  /*!
   * \brief Remove the given sequence from the cache.
   * This includes erasing the sequence from data structures like page table,
   * seq_lengths, etc.
   * The id of all sequences on behind of it will be decreased by 1.
   * \param seq_id The sequence to remove.
   */
  void Remove(int64_t seq_id) {
    CHECK_LT(seq_id, num_total_seqs_);
    for (int32_t page_id : page_table_[seq_id]) {
      FreePage(page_id);
    }
    page_table_.erase(page_table_.begin() + seq_id);
    seq_lengths_.erase(seq_lengths_.begin() + seq_id);
    cur_append_lengths_.erase(cur_append_lengths_.begin() + seq_id);
    --num_total_seqs_;
    dirty_aux_data_device_ = true;
  }

  /*!
   * \brief Pop the last `n` slots of K/V values for the given sequence.
   * \param seq_id The sequence to be processed.
   * \param n The length to pop.
   */
  void PopN(int64_t seq_id, int64_t n) {
    CHECK_LT(seq_id, num_total_seqs_);
    CHECK_GE(n, 0);
    CHECK_LE(n, seq_lengths_[seq_id]);

    // NOTE: this method does not free pages.
    seq_lengths_[seq_id] -= n;
    dirty_aux_data_device_ = true;
  }

  /*!
   * \brief Returning the cached K/V values in the form of "an array of NDArray".
   * Each returned NDArray has layout (num_layers, 2, seqlen, num_heads, head_dim),
   * where along on the "2" dimension, index 0 stands for K and 1 stands for V.
   * \param f_copy_data The function used for copying data out from cache.
   * \return The cached K/V values, one NDArray per sequence.
   * \note This method is majorly for debug and testing purpose.
   */
  Array<NDArray> DebugGetKV(PackedFunc f_copy_data) {
    // The auxiliary data structure on device must have been synchronized.
    CHECK(!dirty_aux_data_device_)
        << "The auxiliary arrays are not synchronized to device. Please call "
           "`SyncAuxArrayToDevice` to synchronize before calling `View`.";

    Array<NDArray> kv_values;
    kv_values.reserve(num_total_seqs_);

    for (int64_t seq_id = 0; seq_id < num_total_seqs_; ++seq_id) {
      NDArray values = NDArray::Empty({num_layers_, 2, num_heads_, seq_lengths_[seq_id], head_dim_},
                                      pages_->dtype, pages_->device);
      f_copy_data(pages_, page_table_indptr_view_, page_table_values_view_, values, seq_id);
      kv_values.push_back(values);
    }
    return kv_values;
  }

  /*! \brief Reset the values in cur_append_lengths to zeros */
  void ResetAppendLengths() {
    ICHECK_EQ(cur_append_lengths_.size(), num_total_seqs_);
    for (int64_t seq_id = 0; seq_id < num_total_seqs_; ++seq_id) {
      cur_append_lengths_[seq_id] = 0;
    }
    dirty_aux_data_device_ = true;
  }

  /*!
   * \brief Synchronize auxiliary arrays to device. The arrays include
   * - page table (represented in CSR format on device),
   * - the number of used slots in the last page of each sequence,
   * - the lengths of append for each sequence,
   * - the position-to-seqid array of each position in the appended data.
   * \note This method resets the dirty flag to false, and needs to be
   * invoked before running any computation on device (attention/append/...).
   */
  void SyncAuxArrayToDevice() {
    int64_t nbyte_aux = (dtype_aux_.bits * dtype_aux_.lanes + 7) / 8;

    // - Invariant checks
    ICHECK_EQ(page_table_.size(), num_total_seqs_);
    ICHECK_EQ(seq_lengths_.size(), num_total_seqs_);
    for (int64_t seq_id = 0; seq_id < num_total_seqs_; ++seq_id) {
      ICHECK(!page_table_[seq_id].empty());
    }
    // - Grow NDArrays when needed.
    DeviceAuxNDArrayGrow();

    // - Copy page table indptr and values and copy to device.
    std::vector<int32_t> page_table_indptr_host = {0};
    std::vector<int32_t> page_table_values_host;
    int64_t npage_in_use = 0;
    for (int64_t seq_id = 0; seq_id < num_total_seqs_; ++seq_id) {
      // NOTE: The page table on host may contain the pages that are not
      // in use by the sequence.
      // Here we only copy the pages in use.
      std::vector<int32_t> seq_page_table = page_table_[seq_id];
      int64_t npage = (seq_lengths_[seq_id] + page_size_ - 1) / page_size_;
      ICHECK_LE(npage, static_cast<int64_t>(seq_page_table.size()));
      page_table_values_host.insert(page_table_values_host.end(), seq_page_table.begin(),
                                    seq_page_table.begin() + npage);
      page_table_indptr_host.push_back(page_table_values_host.size());
      npage_in_use += seq_page_table.size();
    }

    ICHECK_EQ(npage_in_use, num_pages_in_use_);
    ICHECK_EQ(page_table_indptr_host.size(), num_total_seqs_ + 1);
    page_table_indptr_view_ =
        page_table_indptr_device_.CreateView({num_total_seqs_ + 1}, dtype_aux_);
    page_table_indptr_view_.CopyFromBytes(page_table_indptr_host.data(),
                                          (num_total_seqs_ + 1) * nbyte_aux);
    page_table_values_view_ = page_table_values_device_.CreateView(
        {static_cast<int64_t>(page_table_values_host.size())}, dtype_aux_);
    page_table_values_view_.CopyFromBytes(page_table_values_host.data(),
                                          page_table_values_host.size() * nbyte_aux);

    // - Compute `last_page_offset` from seq_lengths and copy to device.
    std::vector<int32_t> last_page_offset_host;
    last_page_offset_host.reserve(num_total_seqs_);
    for (int32_t len : seq_lengths_) {
      ICHECK_GT(len, 0);
      last_page_offset_host.push_back((len - 1) % page_size_ + 1);
    }
    last_page_offset_view_ = last_page_offset_device_.CreateView({num_total_seqs_}, dtype_aux_);
    last_page_offset_view_.CopyFromBytes(last_page_offset_host.data(), num_total_seqs_ * nbyte_aux);

    // - Compute append_length_indptr, pos2seqid and copy to device.
    std::vector<int32_t> append_length_indptr = {0};
    std::vector<int32_t> pos2seqid;
    append_length_indptr.reserve(num_total_seqs_ + 1);

    for (int64_t seq_id = 0; seq_id < num_total_seqs_; ++seq_id) {
      append_length_indptr.push_back(append_length_indptr.back() + cur_append_lengths_[seq_id]);
      for (int64_t pos = 0; pos < cur_append_lengths_[seq_id]; ++pos) {
        pos2seqid.push_back(seq_id);
      }
    }
    CHECK_EQ(append_length_indptr.back(), pos2seqid.size());
    ICHECK_EQ(append_length_indptr.size(), num_total_seqs_ + 1);
    cur_append_length_indptr_view_ =
        cur_append_length_indptr_device_.CreateView({num_total_seqs_ + 1}, dtype_aux_);
    cur_append_length_indptr_view_.CopyFromBytes(append_length_indptr.data(),
                                                 (num_total_seqs_ + 1) * nbyte_aux);
    cur_pos2seqid_view_ =
        cur_pos2seqid_device_.CreateView({static_cast<int64_t>(pos2seqid.size())}, dtype_aux_);
    cur_pos2seqid_view_.CopyFromBytes(pos2seqid.data(), pos2seqid.size() * nbyte_aux);

    // - Reset the dirty flag to false.
    dirty_aux_data_device_ = false;
  }

  /*! \brief Return the number of remaining pages. */
  int GetNumAvailablePages() {
    ICHECK_EQ(num_pages_allocated_, free_page_ids_.size() + num_pages_in_use_);
    return pages_->shape[0] - num_pages_in_use_;
  }

  /*! \brief Reset the KV cache. */
  void Clear() {
    num_total_seqs_ = 0;
    num_pages_in_use_ = 0;
    num_pages_allocated_ = 0;

    free_page_ids_.clear();
    page_table_.clear();
    seq_lengths_.clear();
    cur_append_lengths_.clear();

    dirty_aux_data_device_ = false;
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.vm.PagedAttentionKVCache";
  TVM_DECLARE_FINAL_OBJECT_INFO(PagedAttentionKVCacheObj, Object);

 private:
  /*!
   * \brief Allocate a new page for the given sequence.
   * This function updates the page table for the given sequence.
   */
  void AllocatePageForSequence(int64_t seq_id) {
    ICHECK_LT(seq_id, num_total_seqs_);
    int32_t page_id = GetFreePage();
    page_table_[seq_id].push_back(page_id);
    ++num_pages_in_use_;
  }

  /*! \brief Get a new free page and return its id. */
  int32_t GetFreePage() {
    // Find a page from the free page pools.
    if (!free_page_ids_.empty()) {
      int32_t page_id = free_page_ids_.back();
      free_page_ids_.pop_back();
      return page_id;
    }

    // Allocate a new page.
    int64_t reserved_num_pages = pages_->shape[0];
    if (num_pages_allocated_ < reserved_num_pages) {
      return num_pages_allocated_++;
    }
    CHECK(allow_growth_)
        << "The page KV cache is full and growth is not allowed. Please set a larger "
           "total token capacity when initialization.";
    ICHECK_EQ(num_pages_allocated_, reserved_num_pages);

    // Grow the `pages` array by doubling its size.
    ICHECK_EQ(pages_->ndim, 6);
    std::vector<int64_t> new_shape(pages_->shape, pages_->shape + 6);
    new_shape[0] = reserved_num_pages * 2;
    DLDataType dtype = pages_->dtype;
    NDArray new_pages = NDArray::Empty(new_shape, dtype, pages_->device);
    new_pages.CreateView(pages_.Shape(), dtype).CopyFrom(pages_);
    this->pages_ = new_pages;
    // Also create a larger pos2seqid
    this->cur_pos2seqid_device_ = NDArray::Empty({reserved_num_pages * 2 * page_size_}, dtype_aux_,
                                                 cur_pos2seqid_device_->device);

    return num_pages_allocated_++;
  }

  /*! \brief Free the given page, putting it to the available free page pool. */
  void FreePage(int32_t page_id) {
    free_page_ids_.push_back(page_id);
    --num_pages_in_use_;
  }

  /*! \brief Resize the auxiliary arrays on device as they grow. */
  void DeviceAuxNDArrayGrow() {
    int64_t reserved_nseq = page_table_indptr_device_->shape[0] - 1;
    ICHECK_EQ(last_page_offset_device_->shape[0], reserved_nseq);
    while (num_total_seqs_ > reserved_nseq) {
      reserved_nseq *= 2;
    }

    DLDevice device = page_table_indptr_device_->device;
    if (reserved_nseq != page_table_indptr_device_->shape[0] - 1) {
      CHECK(allow_growth_)
          << "The page KV cache is full and growth is not allowed. Please set a larger "
             "sequence capacity when initialization.";
      page_table_indptr_device_ = NDArray::Empty({reserved_nseq + 1}, dtype_aux_, device);
      last_page_offset_device_ = NDArray::Empty({reserved_nseq}, dtype_aux_, device);
      cur_append_length_indptr_device_ = NDArray::Empty({reserved_nseq + 1}, dtype_aux_, device);
    }

    if (pages_->shape[0] > page_table_values_device_->shape[0]) {
      CHECK(allow_growth_)
          << "The page KV cache is full and growth is not allowed. Please set a larger "
             "total token capacity when initialization.";
      page_table_values_device_ = NDArray::Empty({pages_->shape[0]}, dtype_aux_, device);
    }
  }
};

class PagedAttentionKVCache : public ObjectRef {
 public:
  static PagedAttentionKVCache Create(int64_t reserved_num_seqs, int64_t total_token_capacity,
                                      int64_t page_size, int64_t num_layers, int64_t num_heads,
                                      int64_t head_dim, NDArray init, bool allow_growth) {
    int64_t reserved_num_pages = (total_token_capacity + page_size - 1) / page_size;
    auto n = make_object<PagedAttentionKVCacheObj>(page_size, num_layers, num_heads, head_dim,
                                                   reserved_num_seqs, reserved_num_pages,
                                                   init->dtype, init->device, allow_growth);
    return PagedAttentionKVCache(n);
  }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PagedAttentionKVCache, ObjectRef, PagedAttentionKVCacheObj);
};

TVM_REGISTER_OBJECT_TYPE(PagedAttentionKVCacheObj);

//-------------------------------------------------
//  Register runtime functions
//-------------------------------------------------

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_create")
    .set_body_typed([](ShapeTuple cache_config, int64_t num_layers_, int64_t num_heads_,
                       int64_t head_dim_, NDArray init, bool allow_growth) {
      CHECK_EQ(cache_config.size(), 3);
      return PagedAttentionKVCache::Create(cache_config[0], cache_config[1], cache_config[2],
                                           num_layers_, num_heads_, head_dim_, init, allow_growth);
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_add_sequence")
    .set_body_typed([](PagedAttentionKVCache cache) { return cache->AddSequence(); });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_reserve_extra_length_for_append")
    .set_body_typed([](PagedAttentionKVCache cache, int seq_id, int extra_length) {
      cache->ReserveExtraLengthForAppend(seq_id, extra_length);
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_append")
    .set_body_typed([](PagedAttentionKVCache cache, PackedFunc f_transpose_append, NDArray k_data,
                       NDArray v_data, int64_t layer_id) {
      cache->Append(f_transpose_append, k_data, v_data, layer_id);
      return cache;
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_remove")
    .set_body_typed([](PagedAttentionKVCache cache, int64_t seq_id) { cache->Remove(seq_id); });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_debug_get_kv")
    .set_body_typed([](PagedAttentionKVCache cache, PackedFunc f_view) {
      return cache->DebugGetKV(f_view);
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_popn")
    .set_body_typed([](PagedAttentionKVCache cache, int64_t seq_id, int64_t n) {
      cache->PopN(seq_id, n);
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_reset_append_lengths")
    .set_body_typed([](PagedAttentionKVCache cache) { cache->ResetAppendLengths(); });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_sync_aux_array_to_device")
    .set_body_typed([](PagedAttentionKVCache cache) { cache->SyncAuxArrayToDevice(); });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_clear")
    .set_body_typed([](PagedAttentionKVCache cache) { cache->Clear(); });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_get_num_available_pages")
    .set_body_typed([](PagedAttentionKVCache cache) { return cache->GetNumAvailablePages(); });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_attention")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.size() == 5 || args.size() == 8);
      bool apply_rotary = false;
      double rotary_scale = 1.0;
      double rotary_theta = 1e4;
      if (args.size() == 8) {
        apply_rotary = args[4];
        rotary_scale = args[5];
        rotary_theta = args[6];
      }
      PagedAttentionKVCache cache = args[0];
      cache->Attention(/*f_attention=*/args[1], /*q_data=*/args[2], /*layer_id=*/args[3],
                       /*output=*/args[args.size() - 1], apply_rotary, rotary_scale, rotary_theta);
    });

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
