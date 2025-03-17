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
 * \file src/runtime/relax_vm/attn_backend.h
 * \brief The attention backend classes used by KV cache.
 */

#ifndef TVM_RUNTIME_RELAX_VM_ATTN_BACKEND_H_
#define TVM_RUNTIME_RELAX_VM_ATTN_BACKEND_H_

#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/container/array.h>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "attn_utils.h"

namespace tvm {
namespace runtime {
namespace relax_vm {

/*! \brief The attention backend kinds. */
enum class AttnBackendKind : int {
  kTIR = 0,
  kFlashInfer = 1,
};

/*! \brief The base class of attention backends. */
class AttnBackendFunc {
 public:
  explicit AttnBackendFunc(PackedFunc attn_func, AttnKind attn_kind, AttnBackendKind backend_kind)
      : attn_func_(std::move(attn_func)), attn_kind(attn_kind), backend_kind(backend_kind) {}

  virtual ~AttnBackendFunc() = default;

 protected:
  PackedFunc attn_func_;

 public:
  AttnKind attn_kind;
  AttnBackendKind backend_kind;
};

/*! \brief The paged prefill attention function base class. */
class PagedPrefillFunc : public AttnBackendFunc {
 public:
  explicit PagedPrefillFunc(PackedFunc attn_func, AttnKind attn_kind, AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(int depth, NDArray q, NDArray qo_indptr, NDArray pages, NDArray page_indptr,
                   NDArray page_indices, NDArray length_info, NDArray q_rope_position,
                   NDArray k_rope_pos_offset, bool causal, RoPEMode rope_mode, double rotary_scale,
                   double rotary_theta, double sm_scale, NDArray attn_output, NDArray attn_lse,
                   TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MHA computation is not supported by the current backend";
  }

  virtual void MLA(int depth, NDArray q, NDArray qo_indptr, NDArray pages, NDArray page_indptr,
                   NDArray page_indices, NDArray length_info, bool causal, double sm_scale,
                   NDArray attn_output, NDArray attn_lse, TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MLA computation is not supported by the current backend";
  }

  virtual void BeginForward(int depth, NDArray float_workspace_buffer, NDArray int_workspace_buffer,
                            NDArray page_locked_int_workspace_buffer, HostMemoryVector* qo_indptr,
                            HostMemoryVector* page_indptr, HostMemoryVector* last_page_len,
                            int64_t batch_size, int64_t total_qo_len, int64_t page_size,
                            int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
                            int64_t v_head_dim, bool causal, TVMStreamHandle copy_stream) {
    // Do nothing. Subclasses can override to customize behavior.
  }
};

/*! \brief The TIR-based paged prefill attention function class. */
class TIRPagedPrefillFunc : public PagedPrefillFunc {
 public:
  explicit TIRPagedPrefillFunc(PackedFunc attn_func, AttnKind attn_kind)
      : PagedPrefillFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(int depth, NDArray q, NDArray qo_indptr, NDArray pages, NDArray page_indptr,
           NDArray page_indices, NDArray length_info, NDArray q_rope_position,
           NDArray k_rope_pos_offset, bool causal, RoPEMode rope_mode, double rotary_scale,
           double rotary_theta, double sm_scale, NDArray attn_output, NDArray attn_lse,
           TVMStreamHandle compute_stream) final {
    attn_func_(q, qo_indptr, pages, page_indptr, page_indices, length_info, k_rope_pos_offset,
               q_rope_position, attn_output, attn_lse, static_cast<int64_t>(causal),
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale);
  }

  void MLA(int depth, NDArray q, NDArray qo_indptr, NDArray pages, NDArray page_indptr,
           NDArray page_indices, NDArray length_info, bool causal, double sm_scale,
           NDArray attn_output, NDArray attn_lse, TVMStreamHandle compute_stream) final {
    attn_func_(q, qo_indptr, pages, page_indptr, page_indices, length_info, attn_output, attn_lse,
               static_cast<int64_t>(causal), sm_scale);
  }
};

/*! \brief The FlashInfer-based paged prefill attention function class. */
class FlashInferPagedPrefillFunc : public PagedPrefillFunc {
 public:
  explicit FlashInferPagedPrefillFunc(PackedFunc attn_func, PackedFunc plan_func,
                                      AttnKind attn_kind)
      : PagedPrefillFunc(std::move(attn_func), attn_kind, AttnBackendKind::kFlashInfer),
        plan_func_(std::move(plan_func)) {}

  void MHA(int depth, NDArray q, NDArray qo_indptr, NDArray pages, NDArray page_indptr,
           NDArray page_indices, NDArray length_info, NDArray q_rope_position,
           NDArray k_rope_pos_offset, bool causal, RoPEMode rope_mode, double rotary_scale,
           double rotary_theta, double sm_scale, NDArray attn_output, NDArray attn_lse,
           TVMStreamHandle compute_stream) final {
    auto [float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
          plan_info_vec] = cached_buffers_[depth];
    double rope_rcp_scale = 1 / rotary_scale;
    double rope_rcp_theta = 1 / rotary_theta;
    attn_func_(float_workspace_buffer, int_workspace_buffer, plan_info_vec, q, pages, qo_indptr,
               page_indptr, page_indices, length_info, q_rope_position, k_rope_pos_offset,
               attn_output, attn_lse, /*mask_mode_code=*/static_cast<int64_t>(causal),
               /*pos_encoding_mode_code=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline),
               /*layout(HND)=*/1, /*window_left=*/-1, sm_scale, /*rope_rcp_scale=*/rope_rcp_scale,
               /*rope_rcp_theta=*/rope_rcp_theta, compute_stream);
  }

  void MLA(int depth, NDArray q, NDArray qo_indptr, NDArray pages, NDArray page_indptr,
           NDArray page_indices, NDArray length_info, bool causal, double sm_scale,
           NDArray attn_output, NDArray attn_lse, TVMStreamHandle compute_stream) final {
    auto [float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
          plan_info_vec] = cached_buffers_[depth];
    attn_func_(float_workspace_buffer, int_workspace_buffer, plan_info_vec, q, pages, page_indices,
               attn_output, attn_lse, /*mask_mode_code=*/static_cast<int64_t>(causal),
               /*num_heads=*/q->shape[1], /*page_size=*/pages->shape[1], sm_scale, compute_stream);
  }

  void BeginForward(int depth, NDArray float_workspace_buffer, NDArray int_workspace_buffer,
                    NDArray page_locked_int_workspace_buffer, HostMemoryVector* qo_indptr,
                    HostMemoryVector* page_indptr, HostMemoryVector* last_page_len,
                    int64_t batch_size, int64_t total_qo_len, int64_t page_size,
                    int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
                    int64_t v_head_dim, bool causal, TVMStreamHandle copy_stream) final {
    std::vector<int64_t> kv_len;
    kv_len.reserve(batch_size);
    for (int i = 0; i < static_cast<int>(batch_size); ++i) {
      kv_len.push_back((*page_indptr)[i + 1] != (*page_indptr)[i]
                           ? ((*page_indptr)[i + 1] - (*page_indptr)[i] - 1) * page_size +
                                 (*last_page_len)[i]
                           : 0);
    }
    IntTuple plan_info_vec;
    if (attn_kind == AttnKind::kMHA) {
      // Todo(tvm-team): enable cuda graph
      plan_info_vec = plan_func_(
          float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
          qo_indptr->as_ndarray(), page_indptr->as_ndarray(), IntTuple(std::move(kv_len)),
          total_qo_len, batch_size, num_qo_heads, num_kv_heads, page_size,
          /*enable_cuda_graph=*/false, qk_head_dim, v_head_dim, causal, copy_stream);
    } else if (attn_kind == AttnKind::kMLA) {
      plan_info_vec =
          plan_func_(float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
                     qo_indptr->as_ndarray(), page_indptr->as_ndarray(),
                     IntTuple(std::move(kv_len)), num_qo_heads, v_head_dim, causal, copy_stream);
    }

    if (cached_buffers_.size() <= static_cast<size_t>(depth)) {
      cached_buffers_.resize(depth + 1);
    }
    cached_buffers_[depth] =
        std::make_tuple(float_workspace_buffer, int_workspace_buffer,
                        page_locked_int_workspace_buffer, std::move(plan_info_vec));
  }

 private:
  PackedFunc plan_func_;
  std::vector<std::tuple<NDArray, NDArray, NDArray, IntTuple>> cached_buffers_;
};

/*! \brief The ragged prefill attention function base class. */
class RaggedPrefillFunc : public AttnBackendFunc {
 public:
  explicit RaggedPrefillFunc(PackedFunc attn_func, AttnKind attn_kind, AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(NDArray q, NDArray k, NDArray v, NDArray qo_indptr, NDArray kv_indptr,
                   NDArray q_rope_position, NDArray k_rope_pos_offset, bool causal,
                   RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
                   NDArray attn_output, NDArray attn_lse, TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MHA computation is not supported by the current backend";
  }

  virtual void BeginForward(NDArray float_workspace_buffer, NDArray int_workspace_buffer,
                            NDArray page_locked_int_workspace_buffer, HostMemoryVector* qo_indptr,
                            HostMemoryVector* kv_indptr, int64_t batch_size, int64_t total_qo_len,
                            int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
                            int64_t v_head_dim, bool causal, TVMStreamHandle copy_stream) {
    // Do nothing. Subclasses can override to customize behavior.
  }
};

/*! \brief The TIR-based ragged prefill attention function class. */
class TIRRaggedPrefillFunc : public RaggedPrefillFunc {
 public:
  explicit TIRRaggedPrefillFunc(PackedFunc attn_func, AttnKind attn_kind)
      : RaggedPrefillFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(NDArray q, NDArray k, NDArray v, NDArray qo_indptr, NDArray kv_indptr,
           NDArray q_rope_position, NDArray k_rope_pos_offset, bool causal, RoPEMode rope_mode,
           double rotary_scale, double rotary_theta, double sm_scale, NDArray attn_output,
           NDArray attn_lse, TVMStreamHandle compute_stream) final {
    attn_func_(q, qo_indptr, k, v, kv_indptr, q_rope_position, k_rope_pos_offset, attn_output,
               attn_lse, static_cast<int64_t>(causal),
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale);
  }
};

/*! \brief The FlashInfer-based ragged prefill attention function class. */
class FlashInferRaggedPrefillFunc : public RaggedPrefillFunc {
 public:
  explicit FlashInferRaggedPrefillFunc(PackedFunc attn_func, PackedFunc plan_func,
                                       AttnKind attn_kind)
      : RaggedPrefillFunc(std::move(attn_func), attn_kind, AttnBackendKind::kFlashInfer),
        plan_func_(std::move(plan_func)) {}

  void MHA(NDArray q, NDArray k, NDArray v, NDArray qo_indptr, NDArray kv_indptr,
           NDArray q_rope_position, NDArray k_rope_pos_offset, bool causal, RoPEMode rope_mode,
           double rotary_scale, double rotary_theta, double sm_scale, NDArray attn_output,
           NDArray attn_lse, TVMStreamHandle compute_stream) final {
    double rope_rcp_scale = 1 / rotary_scale;
    double rope_rcp_theta = 1 / rotary_theta;
    attn_func_(float_workspace_buffer_, int_workspace_buffer_, plan_info_vec_, q, k, v, qo_indptr,
               kv_indptr, q_rope_position, k_rope_pos_offset, attn_output, attn_lse,
               /*mask_mode_code=*/static_cast<int64_t>(causal),
               /*pos_encoding_mode_code=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline),
               /*layout(NHD)=*/0, /*window_left=*/-1, sm_scale,
               /*rope_rcp_scale=*/rope_rcp_scale,
               /*rope_rcp_theta=*/rope_rcp_theta, compute_stream);
  }

  void BeginForward(NDArray float_workspace_buffer, NDArray int_workspace_buffer,
                    NDArray page_locked_int_workspace_buffer, HostMemoryVector* qo_indptr,
                    HostMemoryVector* kv_indptr, int64_t batch_size, int64_t total_qo_len,
                    int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
                    int64_t v_head_dim, bool causal, TVMStreamHandle copy_stream) final {
    std::vector<int64_t> kv_len;
    kv_len.reserve(batch_size);
    for (int i = 0; i < static_cast<int>(batch_size); ++i) {
      kv_len.push_back((*kv_indptr)[i + 1] - (*kv_indptr)[i]);
    }
    // Todo(tvm-team): enable cuda graph
    float_workspace_buffer_ = float_workspace_buffer;
    int_workspace_buffer_ = int_workspace_buffer;
    page_locked_int_workspace_buffer_ = page_locked_int_workspace_buffer;
    plan_info_vec_ =
        plan_func_(float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
                   qo_indptr->as_ndarray(), kv_indptr->as_ndarray(), IntTuple(std::move(kv_len)),
                   total_qo_len, batch_size, num_qo_heads, num_kv_heads, /*page_size=*/1,
                   /*enable_cuda_graph=*/false, qk_head_dim, v_head_dim, causal, copy_stream);
  }

 private:
  PackedFunc plan_func_;
  NDArray float_workspace_buffer_;
  NDArray int_workspace_buffer_;
  NDArray page_locked_int_workspace_buffer_;
  IntTuple plan_info_vec_;
};

/*! \brief The paged decode attention function base class. */
class PagedDecodeFunc : public AttnBackendFunc {
 public:
  explicit PagedDecodeFunc(PackedFunc attn_func, AttnKind attn_kind, AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(int depth, NDArray q, NDArray pages, NDArray page_indptr, NDArray page_indices,
                   NDArray length_info, NDArray k_rope_pos_offset, NDArray q_rope_position,
                   RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
                   NDArray attn_output, NDArray attn_lse, TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MHA computation is not supported by the current backend";
  }

  virtual void MLA(int depth, NDArray q, NDArray pages, NDArray page_indptr, NDArray page_indices,
                   NDArray length_info, double sm_scale, NDArray attn_output, NDArray attn_lse,
                   TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MLA computation is not supported by the current backend";
  }

  virtual void BeginForward(int depth, NDArray float_workspace_buffer, NDArray int_workspace_buffer,
                            NDArray page_locked_int_workspace_buffer, HostMemoryVector* page_indptr,
                            int64_t batch_size, int64_t page_size, int64_t num_qo_heads,
                            int64_t num_kv_heads, int64_t qk_head_dim, int64_t v_head_dim,
                            RoPEMode rope_mode, DataType q_dtype, DataType kv_dtype,
                            TVMStreamHandle copy_stream) {
    // Do nothing. Subclasses can override to customize behavior.
  }
};

/*! \brief The TIR-based paged decode attention function class. */
class TIRPagedDecodeFunc : public PagedDecodeFunc {
 public:
  explicit TIRPagedDecodeFunc(PackedFunc attn_func, AttnKind attn_kind)
      : PagedDecodeFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(int depth, NDArray q, NDArray pages, NDArray page_indptr, NDArray page_indices,
           NDArray length_info, NDArray k_rope_pos_offset, NDArray q_rope_position,
           RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
           NDArray attn_output, NDArray attn_lse, TVMStreamHandle compute_stream) final {
    attn_func_(q, pages, page_indptr, page_indices, length_info, k_rope_pos_offset, q_rope_position,
               attn_output, attn_lse,
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale);
  }

  void MLA(int depth, NDArray q, NDArray pages, NDArray page_indptr, NDArray page_indices,
           NDArray length_info, double sm_scale, NDArray attn_output, NDArray attn_lse,
           TVMStreamHandle compute_stream) final {
    attn_func_(q, pages, page_indptr, page_indices, length_info, attn_output, attn_lse, sm_scale);
  }
};

/*! \brief The FlashInfer-based paged decode attention function class. */
class FlashInferPagedDecodeFunc : public PagedDecodeFunc {
 public:
  explicit FlashInferPagedDecodeFunc(PackedFunc attn_func, PackedFunc plan_func, AttnKind attn_kind)
      : PagedDecodeFunc(std::move(attn_func), attn_kind, AttnBackendKind::kFlashInfer),
        plan_func_(std::move(plan_func)) {}

  void MHA(int depth, NDArray q, NDArray pages, NDArray page_indptr, NDArray page_indices,
           NDArray length_info, NDArray k_rope_pos_offset, NDArray q_rope_position,
           RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
           NDArray attn_output, NDArray attn_lse, TVMStreamHandle compute_stream) final {
    auto [float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
          plan_info_vec] = cached_buffers_[depth];
    double rope_rcp_scale = 1 / rotary_scale;
    double rope_rcp_theta = 1 / rotary_theta;
    attn_func_(float_workspace_buffer, int_workspace_buffer, plan_info_vec, q, pages, page_indptr,
               page_indices, length_info, q_rope_position, k_rope_pos_offset, attn_output, attn_lse,
               /*pos_encoding_mode_code=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline),
               /*layout(HND)=*/1, /*window_left=*/-1, sm_scale, /*rope_rcp_scale=*/rope_rcp_scale,
               /*rope_rcp_theta=*/rope_rcp_theta, compute_stream);
  }

  void BeginForward(int depth, NDArray float_workspace_buffer, NDArray int_workspace_buffer,
                    NDArray page_locked_int_workspace_buffer, HostMemoryVector* page_indptr,
                    int64_t batch_size, int64_t page_size, int64_t num_qo_heads,
                    int64_t num_kv_heads, int64_t qk_head_dim, int64_t v_head_dim,
                    RoPEMode rope_mode, DataType q_dtype, DataType kv_dtype,
                    TVMStreamHandle copy_stream) final {
    // Todo(tvm-team): enable cuda graph
    IntTuple plan_info_vec = plan_func_(
        float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
        page_indptr->as_ndarray(), batch_size, num_qo_heads, num_kv_heads, page_size,
        /*enable_cuda_graph=*/false, static_cast<int64_t>(rope_mode == RoPEMode::kInline),
        /*window_left=*/-1, qk_head_dim, v_head_dim, q_dtype, kv_dtype, copy_stream);

    if (cached_buffers_.size() <= static_cast<size_t>(depth)) {
      cached_buffers_.resize(depth + 1);
    }
    cached_buffers_[depth] =
        std::make_tuple(float_workspace_buffer, int_workspace_buffer,
                        page_locked_int_workspace_buffer, std::move(plan_info_vec));
  }

 private:
  PackedFunc plan_func_;
  std::vector<std::tuple<NDArray, NDArray, NDArray, IntTuple>> cached_buffers_;
};

/*! \brief The paged prefill with tree mask attention function base class. */
class PagedPrefillTreeMaskFunc : public AttnBackendFunc {
 public:
  explicit PagedPrefillTreeMaskFunc(PackedFunc attn_func, AttnKind attn_kind,
                                    AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(NDArray q, NDArray qo_indptr, NDArray pages, NDArray page_indptr,
                   NDArray page_indices, NDArray length_info, NDArray k_rope_pos_offset,
                   NDArray q_rope_position, NDArray tree_attn_mn_indptr, NDArray tree_attn_mask,
                   RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
                   NDArray attn_output, NDArray attn_lse, TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MHA computation is not supported by the current backend";
  }

  virtual void MLA(NDArray q, NDArray qo_indptr, NDArray pages, NDArray page_indptr,
                   NDArray page_indices, NDArray length_info, NDArray tree_attn_mn_indptr,
                   NDArray tree_attn_mask, double sm_scale, NDArray attn_output, NDArray attn_lse,
                   TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MLA computation is not supported by the current backend";
  }

  virtual void BeginForward(NDArray temp_float_attn_workspace, NDArray temp_int_attn_workspace,
                            HostMemoryVector* page_indptr, HostMemoryVector* last_page_len,
                            HostMemoryVector* qo_indptr, int64_t batch_size, int64_t page_size,
                            int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
                            int64_t v_head_dim, RoPEMode rope_mode, TVMStreamHandle copy_stream) {
    // Do nothing. Subclasses can override to customize behavior.
  }
};

/*! \brief The TIR-based paged prefill with tree mask attention function class. */
class TIRPagedPrefillTreeMaskFunc : public PagedPrefillTreeMaskFunc {
 public:
  explicit TIRPagedPrefillTreeMaskFunc(PackedFunc attn_func, AttnKind attn_kind)
      : PagedPrefillTreeMaskFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(NDArray q, NDArray qo_indptr, NDArray pages, NDArray page_indptr, NDArray page_indices,
           NDArray length_info, NDArray k_rope_pos_offset, NDArray q_rope_position,
           NDArray tree_attn_mn_indptr, NDArray tree_attn_mask, RoPEMode rope_mode,
           double rotary_scale, double rotary_theta, double sm_scale, NDArray attn_output,
           NDArray attn_lse, TVMStreamHandle compute_stream) final {
    attn_func_(q, qo_indptr, pages, page_indptr, page_indices, length_info, k_rope_pos_offset,
               q_rope_position, attn_output, attn_lse,
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale, tree_attn_mn_indptr, tree_attn_mask);
  }
};

/*! \brief The ragged prefill with tree mask function base class. */
class RaggedPrefillTreeMaskFunc : public AttnBackendFunc {
 public:
  explicit RaggedPrefillTreeMaskFunc(PackedFunc attn_func, AttnKind attn_kind,
                                     AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(NDArray q, NDArray k, NDArray v, NDArray qo_indptr, NDArray kv_indptr,
                   NDArray q_rope_position, NDArray tree_attn_mn_indptr, NDArray tree_attn_mask,
                   RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
                   NDArray attn_output, NDArray attn_lse, TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MHA computation is not supported by the current backend";
  }

  virtual void MLA(NDArray q, NDArray compressed_kv, NDArray k_pe, NDArray qo_indptr,
                   NDArray kv_indptr, NDArray tree_attn_mn_indptr, NDArray tree_attn_mask,
                   double sm_scale, NDArray attn_output, NDArray attn_lse,
                   TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MLA computation is not supported by the current backend";
  }

  virtual void BeginForward(NDArray temp_float_attn_workspace, NDArray temp_int_attn_workspace,
                            HostMemoryVector* page_indptr, HostMemoryVector* last_page_len,
                            HostMemoryVector* qo_indptr, int64_t batch_size, int64_t page_size,
                            int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
                            int64_t v_head_dim, RoPEMode rope_mode, TVMStreamHandle copy_stream) {
    // Do nothing. Subclasses can override to customize behavior.
  }
};

/*! \brief The TIR-based ragged prefill with tree mask attention function class. */
class TIRRaggedPrefillTreeMaskFunc : public RaggedPrefillTreeMaskFunc {
 public:
  explicit TIRRaggedPrefillTreeMaskFunc(PackedFunc attn_func, AttnKind attn_kind)
      : RaggedPrefillTreeMaskFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(NDArray q, NDArray k, NDArray v, NDArray qo_indptr, NDArray kv_indptr,
           NDArray q_rope_position, NDArray tree_attn_mn_indptr, NDArray tree_attn_mask,
           RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
           NDArray attn_output, NDArray attn_lse, TVMStreamHandle compute_stream) final {
    attn_func_(q, qo_indptr, k, v, kv_indptr, q_rope_position, tree_attn_mn_indptr, tree_attn_mask,
               attn_output, attn_lse,
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale);
  }
};

/*!
 * \brief Create a PagedPrefillFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention PackedFuncs.
 * \param attn_kind The attention kind of the function.
 * \return The created PagedPrefillFunc pointer.
 */
std::unique_ptr<PagedPrefillFunc> ConvertPagedPrefillFunc(Array<ObjectRef> args,
                                                          AttnKind attn_kind);

/*!
 * \brief Create a PagedDecodeFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention PackedFuncs.
 * \param attn_kind The attention kind of the function.
 * \return The created PagedDecodeFunc pointer.
 */
std::unique_ptr<PagedDecodeFunc> ConvertPagedDecodeFunc(Array<ObjectRef> args, AttnKind attn_kind);

/*!
 * \brief Create a RaggedPrefillFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention PackedFuncs.
 * \param attn_kind The attention kind of the function.
 * \return The created RaggedPrefillFunc pointer.
 */
std::unique_ptr<RaggedPrefillFunc> ConvertRaggedPrefillFunc(Array<ObjectRef> args,
                                                            AttnKind attn_kind);

/*!
 * \brief Create a PagedPrefillTreeMaskFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention PackedFuncs.
 * \param attn_kind The attention kind of the function.
 * \return The created PagedPrefillTreeMaskFunc pointer.
 */
std::unique_ptr<PagedPrefillTreeMaskFunc> ConvertPagedPrefillTreeMaskFunc(Array<ObjectRef> args,
                                                                          AttnKind attn_kind);

/*!
 * \brief Create a RaggedPrefillTreeMaskFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention PackedFuncs.
 * \param attn_kind The attention kind of the function.
 * \return The created RaggedPrefillTreeMaskFunc pointer.
 */
std::unique_ptr<RaggedPrefillTreeMaskFunc> ConvertRaggedPrefillTreeMaskFunc(Array<ObjectRef> args,
                                                                            AttnKind attn_kind);

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_RELAX_VM_ATTN_BACKEND_H_
