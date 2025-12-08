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
 * \file src/runtime/vm/attn_backend.h
 * \brief The attention backend classes used by KV cache.
 */

#ifndef TVM_RUNTIME_VM_ATTN_BACKEND_H_
#define TVM_RUNTIME_VM_ATTN_BACKEND_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/int_tuple.h>
#include <tvm/runtime/logging.h>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "attn_utils.h"

namespace tvm {
namespace runtime {
namespace vm {

/*! \brief The attention backend kinds. */
enum class AttnBackendKind : int {
  kTIR = 0,
  kFlashInfer = 1,
};

/*! \brief The base class of attention backends. */
class AttnBackendFunc {
 public:
  explicit AttnBackendFunc(ffi::Function attn_func, AttnKind attn_kind,
                           AttnBackendKind backend_kind)
      : attn_func_(std::move(attn_func)), attn_kind(attn_kind), backend_kind(backend_kind) {}

  virtual ~AttnBackendFunc() = default;

 protected:
  // helper allocator class for creating strided view of a Tensor
  // that applies byte offset to the original data pointer
  class ViewBasedAlloc {
   public:
    explicit ViewBasedAlloc(Tensor source) : source_(source) {}
    void AllocData(DLTensor* tensor, int64_t* strides, int64_t extra_byte_offset) {
      tensor->data = static_cast<char*>(source_->data) + extra_byte_offset;
      tensor->strides = strides;
    }

    void FreeData(DLTensor* tensor) {}

   private:
    Tensor source_;
  };

  ffi::Function attn_func_;

 public:
  AttnKind attn_kind;
  AttnBackendKind backend_kind;
};

/*! \brief The paged prefill attention function base class. */
class PagedPrefillFunc : public AttnBackendFunc {
 public:
  explicit PagedPrefillFunc(ffi::Function attn_func, AttnKind attn_kind,
                            AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(int depth, Tensor q, Tensor qo_indptr, Tensor pages, Tensor page_indptr,
                   Tensor page_indices, Tensor length_info, Tensor q_rope_position,
                   Tensor k_rope_pos_offset, bool causal, RoPEMode rope_mode, double rotary_scale,
                   double rotary_theta, double sm_scale, Tensor attn_output, Tensor attn_lse,
                   TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MHA computation is not supported by the current backend";
  }

  virtual void MLA(int depth, Tensor q, Tensor qo_indptr, Tensor pages, Tensor page_indptr,
                   Tensor page_indices, Tensor length_info, bool causal, double sm_scale,
                   Tensor attn_output, Tensor attn_lse, TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MLA computation is not supported by the current backend";
  }

  virtual void BeginForward(int depth, Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                            Tensor page_locked_int_workspace_buffer, HostMemoryVector* qo_indptr,
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
  explicit TIRPagedPrefillFunc(ffi::Function attn_func, AttnKind attn_kind)
      : PagedPrefillFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(int depth, Tensor q, Tensor qo_indptr, Tensor pages, Tensor page_indptr,
           Tensor page_indices, Tensor length_info, Tensor q_rope_position,
           Tensor k_rope_pos_offset, bool causal, RoPEMode rope_mode, double rotary_scale,
           double rotary_theta, double sm_scale, Tensor attn_output, Tensor attn_lse,
           TVMStreamHandle compute_stream) final {
    attn_func_(q, qo_indptr, pages, page_indptr, page_indices, length_info, k_rope_pos_offset,
               q_rope_position, attn_output, attn_lse, static_cast<int64_t>(causal),
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale);
  }

  void MLA(int depth, Tensor q, Tensor qo_indptr, Tensor pages, Tensor page_indptr,
           Tensor page_indices, Tensor length_info, bool causal, double sm_scale,
           Tensor attn_output, Tensor attn_lse, TVMStreamHandle compute_stream) final {
    attn_func_(q, qo_indptr, pages, page_indptr, page_indices, length_info, attn_output, attn_lse,
               static_cast<int64_t>(causal), sm_scale);
  }
};

/*! \brief The FlashInfer-based paged prefill attention function class. */
class FlashInferPagedPrefillFunc : public PagedPrefillFunc {
 public:
  explicit FlashInferPagedPrefillFunc(ffi::Function attn_func, ffi::Function plan_func,
                                      AttnKind attn_kind)
      : PagedPrefillFunc(std::move(attn_func), attn_kind, AttnBackendKind::kFlashInfer),
        plan_func_(std::move(plan_func)) {}

  void MHA(int depth, Tensor q, Tensor qo_indptr, Tensor pages, Tensor page_indptr,
           Tensor page_indices, Tensor length_info, Tensor q_rope_position,
           Tensor k_rope_pos_offset, bool causal, RoPEMode rope_mode, double rotary_scale,
           double rotary_theta, double sm_scale, Tensor attn_output, Tensor attn_lse,
           TVMStreamHandle compute_stream) final {
    Device device = q->device;
    TVMStreamHandle original_stream = DeviceAPI::Get(device)->GetCurrentStream(device);
    DeviceAPI::Get(device)->SetStream(device, compute_stream);
    auto [float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
          plan_info_vec] = cached_buffers_[depth];
    double rope_rcp_scale = 1 / rotary_scale;
    double rope_rcp_theta = 1 / rotary_theta;

    ICHECK_EQ(pages.ndim(), 5);
    int H = pages->shape[2];
    int N = pages->shape[3];
    int D = pages->shape[4];
    CHECK(pages.IsContiguous());
    std::vector<int64_t> pages_k_v_shape = {pages->shape[0], H, N, D};
    std::vector<int64_t> pages_k_v_strides = {2 * H * N * D, N * D, D, 1};
    Tensor pages_k =
        Tensor::FromNDAlloc(ViewBasedAlloc(pages), ffi::Shape(pages_k_v_shape), pages->dtype,
                            pages->device, pages_k_v_strides.data(), pages->byte_offset);
    Tensor pages_v = Tensor::FromNDAlloc(
        ViewBasedAlloc(pages), ffi::Shape(pages_k_v_shape), pages->dtype, pages->device,
        pages_k_v_strides.data(), pages->byte_offset + (H * N * D) * pages.DataType().bytes());

    attn_func_(float_workspace_buffer, int_workspace_buffer, plan_info_vec, q, pages_k, pages_v,
               qo_indptr, page_indptr, page_indices, length_info, attn_output, attn_lse,
               /*mask_mode_code=*/static_cast<int64_t>(causal), /*layout(HND)=*/1,
               /*window_left=*/-1, /*enable_pdl=*/false, sm_scale,
               /*rope_rcp_scale=*/rope_rcp_scale, /*rope_rcp_theta=*/rope_rcp_theta);
    DeviceAPI::Get(device)->SetStream(device, original_stream);
  }

  void MLA(int depth, Tensor q, Tensor qo_indptr, Tensor pages, Tensor page_indptr,
           Tensor page_indices, Tensor length_info, bool causal, double sm_scale,
           Tensor attn_output, Tensor attn_lse, TVMStreamHandle compute_stream) final {
    auto [float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
          plan_info_vec] = cached_buffers_[depth];
    Device device = q->device;
    TVMStreamHandle original_stream = DeviceAPI::Get(device)->GetCurrentStream(device);
    DeviceAPI::Get(device)->SetStream(device, compute_stream);
    ICHECK_NE(qk_head_dim_, -1);
    ICHECK_NE(v_head_dim_, -1);
    int64_t H = q->shape[1];
    int64_t page_size = pages->shape[1];
    int64_t rope_head_dim = qk_head_dim_ - v_head_dim_;
    int64_t nope_head_dim = q->shape[2] - rope_head_dim;

    // Split q into q_nope and q_pe
    CHECK(q.IsContiguous());
    std::vector<int64_t> q_nope_shape = {q->shape[0], H, nope_head_dim};
    std::vector<int64_t> q_pe_shape = {q->shape[0], H, rope_head_dim};
    std::vector<int64_t> q_strides = {H * q->shape[2], q->shape[2], 1};
    Tensor q_nope = Tensor::FromNDAlloc(ViewBasedAlloc(q), ffi::Shape(q_nope_shape), q->dtype,
                                        q->device, q_strides.data(), q->byte_offset);
    Tensor q_pe = Tensor::FromNDAlloc(ViewBasedAlloc(q), ffi::Shape(q_pe_shape), q->dtype,
                                      q->device, q_strides.data(),
                                      q->byte_offset + nope_head_dim * q.DataType().bytes());
    // Split pages into kv_nope and kv_pe
    CHECK(pages.IsContiguous());
    std::vector<int64_t> kv_nope_shape = {pages->shape[0], page_size, nope_head_dim};
    std::vector<int64_t> kv_pe_shape = {pages->shape[0], page_size, rope_head_dim};
    std::vector<int64_t> kv_strides = {page_size * pages->shape[2], pages->shape[2], 1};
    Tensor kv_nope =
        Tensor::FromNDAlloc(ViewBasedAlloc(pages), ffi::Shape(kv_nope_shape), pages->dtype,
                            pages->device, kv_strides.data(), pages->byte_offset);
    Tensor kv_pe = Tensor::FromNDAlloc(
        ViewBasedAlloc(pages), ffi::Shape(kv_pe_shape), pages->dtype, pages->device,
        kv_strides.data(), pages->byte_offset + nope_head_dim * pages.DataType().bytes());

    attn_func_(float_workspace_buffer, int_workspace_buffer, plan_info_vec, q_nope, q_pe, kv_nope,
               kv_pe, page_indices, attn_output, attn_lse,
               /*mask_mode_code=*/static_cast<int64_t>(causal),
               /*num_heads=*/q->shape[1], /*page_size=*/pages->shape[1], sm_scale);
    DeviceAPI::Get(device)->SetStream(device, original_stream);
  }

  void BeginForward(int depth, Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                    Tensor page_locked_int_workspace_buffer, HostMemoryVector* qo_indptr,
                    HostMemoryVector* page_indptr, HostMemoryVector* last_page_len,
                    int64_t batch_size, int64_t total_qo_len, int64_t page_size,
                    int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
                    int64_t v_head_dim, bool causal, TVMStreamHandle copy_stream) final {
    Tensor kv_len_arr = Tensor::Empty({batch_size}, DataType::Int(32), Device{kDLCPU, 0});
    int32_t* kv_len_arr_data = static_cast<int32_t*>(kv_len_arr.data_ptr());
    for (int i = 0; i < static_cast<int>(batch_size); ++i) {
      kv_len_arr_data[i] =
          (*page_indptr)[i + 1] != (*page_indptr)[i]
              ? ((*page_indptr)[i + 1] - (*page_indptr)[i] - 1) * page_size + (*last_page_len)[i]
              : 0;
    }
    qk_head_dim_ = qk_head_dim;
    v_head_dim_ = v_head_dim;
    ffi::Array<int64_t> plan_info_vec;
    Device device = float_workspace_buffer->device;
    TVMStreamHandle original_stream = DeviceAPI::Get(device)->GetCurrentStream(device);
    DeviceAPI::Get(device)->SetStream(device, copy_stream);
    if (attn_kind == AttnKind::kMHA) {
      // Todo(tvm-team): enable cuda graph
      plan_info_vec =
          plan_func_(float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
                     qo_indptr->as_tensor(), page_indptr->as_tensor(), kv_len_arr, total_qo_len,
                     batch_size, num_qo_heads, num_kv_heads, page_size,
                     /*enable_cuda_graph=*/false, qk_head_dim, v_head_dim, causal,
                     /*window_left=*/-1, /*fixed_split_size=*/-1, /*disable_split_kv=*/false,
                     /*num_colocated_ctas=*/0)
              .cast<ffi::Array<int64_t>>();
    } else if (attn_kind == AttnKind::kMLA) {
      plan_info_vec =
          plan_func_(float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
                     qo_indptr->as_tensor(), page_indptr->as_tensor(), kv_len_arr, num_qo_heads,
                     v_head_dim, causal)
              .cast<ffi::Array<int64_t>>();
    }
    DeviceAPI::Get(device)->SetStream(device, original_stream);

    if (cached_buffers_.size() <= static_cast<size_t>(depth)) {
      cached_buffers_.resize(depth + 1);
    }
    cached_buffers_[depth] =
        std::make_tuple(float_workspace_buffer, int_workspace_buffer,
                        page_locked_int_workspace_buffer, std::move(plan_info_vec));
  }

 private:
  int64_t qk_head_dim_ = -1;
  int64_t v_head_dim_ = -1;
  ffi::Function plan_func_;
  std::vector<std::tuple<Tensor, Tensor, Tensor, ffi::Array<int64_t>>> cached_buffers_;
};

/*! \brief The ragged prefill attention function base class. */
class RaggedPrefillFunc : public AttnBackendFunc {
 public:
  explicit RaggedPrefillFunc(ffi::Function attn_func, AttnKind attn_kind,
                             AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(Tensor q, Tensor k, Tensor v, Tensor qo_indptr, Tensor kv_indptr,
                   Tensor q_rope_position, Tensor k_rope_pos_offset, bool causal,
                   RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
                   Tensor attn_output, Tensor attn_lse, TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MHA computation is not supported by the current backend";
  }

  virtual void BeginForward(Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                            Tensor page_locked_int_workspace_buffer, HostMemoryVector* qo_indptr,
                            HostMemoryVector* kv_indptr, int64_t batch_size, int64_t total_qo_len,
                            int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
                            int64_t v_head_dim, bool causal, TVMStreamHandle copy_stream) {
    // Do nothing. Subclasses can override to customize behavior.
  }
};

/*! \brief The TIR-based ragged prefill attention function class. */
class TIRRaggedPrefillFunc : public RaggedPrefillFunc {
 public:
  explicit TIRRaggedPrefillFunc(ffi::Function attn_func, AttnKind attn_kind)
      : RaggedPrefillFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(Tensor q, Tensor k, Tensor v, Tensor qo_indptr, Tensor kv_indptr, Tensor q_rope_position,
           Tensor k_rope_pos_offset, bool causal, RoPEMode rope_mode, double rotary_scale,
           double rotary_theta, double sm_scale, Tensor attn_output, Tensor attn_lse,
           TVMStreamHandle compute_stream) final {
    attn_func_(q, qo_indptr, k, v, kv_indptr, q_rope_position, k_rope_pos_offset, attn_output,
               attn_lse, static_cast<int64_t>(causal),
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale);
  }
};

/*! \brief The FlashInfer-based ragged prefill attention function class. */
class FlashInferRaggedPrefillFunc : public RaggedPrefillFunc {
 public:
  explicit FlashInferRaggedPrefillFunc(ffi::Function attn_func, ffi::Function plan_func,
                                       AttnKind attn_kind, int64_t qk_head_dim_override,
                                       int64_t v_head_dim_override)
      : RaggedPrefillFunc(std::move(attn_func), attn_kind, AttnBackendKind::kFlashInfer),
        qk_head_dim_override_(qk_head_dim_override),
        v_head_dim_override_(v_head_dim_override),
        plan_func_(std::move(plan_func)) {}

  void MHA(Tensor q, Tensor k, Tensor v, Tensor qo_indptr, Tensor kv_indptr, Tensor q_rope_position,
           Tensor k_rope_pos_offset, bool causal, RoPEMode rope_mode, double rotary_scale,
           double rotary_theta, double sm_scale, Tensor attn_output, Tensor attn_lse,
           TVMStreamHandle compute_stream) final {
    Device device = q->device;
    TVMStreamHandle original_stream = DeviceAPI::Get(device)->GetCurrentStream(device);
    DeviceAPI::Get(device)->SetStream(device, compute_stream);
    double rope_rcp_scale = 1 / rotary_scale;
    double rope_rcp_theta = 1 / rotary_theta;
    attn_func_(float_workspace_buffer_, int_workspace_buffer_, plan_info_vec_, q, k, v, qo_indptr,
               kv_indptr, attn_output, attn_lse,
               /*mask_mode_code=*/static_cast<int64_t>(causal),
               /*layout(NHD)=*/0, /*window_left=*/-1,
               /*enable_pdl=*/false, sm_scale,
               /*rope_rcp_scale=*/rope_rcp_scale,
               /*rope_rcp_theta=*/rope_rcp_theta);
    DeviceAPI::Get(device)->SetStream(device, original_stream);
  }

  void BeginForward(Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                    Tensor page_locked_int_workspace_buffer, HostMemoryVector* qo_indptr,
                    HostMemoryVector* kv_indptr, int64_t batch_size, int64_t total_qo_len,
                    int64_t num_qo_heads, int64_t num_kv_heads, int64_t qk_head_dim,
                    int64_t v_head_dim, bool causal, TVMStreamHandle copy_stream) final {
    Tensor kv_len_arr = Tensor::Empty({batch_size}, DataType::Int(32), Device{kDLCPU, 0});
    int32_t* kv_len_arr_data = static_cast<int32_t*>(kv_len_arr.data_ptr());
    for (int i = 0; i < static_cast<int>(batch_size); ++i) {
      kv_len_arr_data[i] = (*kv_indptr)[i + 1] - (*kv_indptr)[i];
    }
    if (qk_head_dim_override_ != -1) {
      qk_head_dim = qk_head_dim_override_;
    }
    if (v_head_dim_override_ != -1) {
      v_head_dim = v_head_dim_override_;
    }
    // Todo(tvm-team): enable cuda graph
    float_workspace_buffer_ = float_workspace_buffer;
    int_workspace_buffer_ = int_workspace_buffer;
    page_locked_int_workspace_buffer_ = page_locked_int_workspace_buffer;
    Device device = float_workspace_buffer->device;
    TVMStreamHandle original_stream = DeviceAPI::Get(device)->GetCurrentStream(device);
    DeviceAPI::Get(device)->SetStream(device, copy_stream);
    plan_info_vec_ =
        plan_func_(float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
                   qo_indptr->as_tensor(), kv_indptr->as_tensor(), kv_len_arr, total_qo_len,
                   batch_size, num_qo_heads, num_kv_heads, /*page_size=*/1,
                   /*enable_cuda_graph=*/false, qk_head_dim, v_head_dim, causal,
                   /*window_left=*/-1, /*fixed_split_size=*/-1, /*disable_split_kv=*/false,
                   /*num_colocated_ctas=*/0)
            .cast<ffi::Array<int64_t>>();
    DeviceAPI::Get(device)->SetStream(device, original_stream);
  }

 private:
  int64_t qk_head_dim_override_;
  int64_t v_head_dim_override_;
  ffi::Function plan_func_;
  Tensor float_workspace_buffer_;
  Tensor int_workspace_buffer_;
  Tensor page_locked_int_workspace_buffer_;
  ffi::Array<int64_t> plan_info_vec_;
};

/*! \brief The paged decode attention function base class. */
class PagedDecodeFunc : public AttnBackendFunc {
 public:
  explicit PagedDecodeFunc(ffi::Function attn_func, AttnKind attn_kind,
                           AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(int depth, Tensor q, Tensor pages, Tensor page_indptr, Tensor page_indices,
                   Tensor length_info, Tensor k_rope_pos_offset, Tensor q_rope_position,
                   RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
                   Tensor attn_output, Tensor attn_lse, TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MHA computation is not supported by the current backend";
  }

  virtual void MLA(int depth, Tensor q, Tensor pages, Tensor page_indptr, Tensor page_indices,
                   Tensor length_info, double sm_scale, Tensor attn_output, Tensor attn_lse,
                   TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MLA computation is not supported by the current backend";
  }

  virtual void BeginForward(int depth, Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                            Tensor page_locked_int_workspace_buffer, HostMemoryVector* page_indptr,
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
  explicit TIRPagedDecodeFunc(ffi::Function attn_func, AttnKind attn_kind)
      : PagedDecodeFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(int depth, Tensor q, Tensor pages, Tensor page_indptr, Tensor page_indices,
           Tensor length_info, Tensor k_rope_pos_offset, Tensor q_rope_position, RoPEMode rope_mode,
           double rotary_scale, double rotary_theta, double sm_scale, Tensor attn_output,
           Tensor attn_lse, TVMStreamHandle compute_stream) final {
    attn_func_(q, pages, page_indptr, page_indices, length_info, k_rope_pos_offset, q_rope_position,
               attn_output, attn_lse,
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale);
  }

  void MLA(int depth, Tensor q, Tensor pages, Tensor page_indptr, Tensor page_indices,
           Tensor length_info, double sm_scale, Tensor attn_output, Tensor attn_lse,
           TVMStreamHandle compute_stream) final {
    attn_func_(q, pages, page_indptr, page_indices, length_info, attn_output, attn_lse, sm_scale);
  }
};

/*! \brief The FlashInfer-based paged decode attention function class. */
class FlashInferPagedDecodeFunc : public PagedDecodeFunc {
 public:
  explicit FlashInferPagedDecodeFunc(ffi::Function attn_func, ffi::Function plan_func,
                                     AttnKind attn_kind)
      : PagedDecodeFunc(std::move(attn_func), attn_kind, AttnBackendKind::kFlashInfer),
        plan_func_(std::move(plan_func)) {}

  void MHA(int depth, Tensor q, Tensor pages, Tensor page_indptr, Tensor page_indices,
           Tensor length_info, Tensor k_rope_pos_offset, Tensor q_rope_position, RoPEMode rope_mode,
           double rotary_scale, double rotary_theta, double sm_scale, Tensor attn_output,
           Tensor attn_lse, TVMStreamHandle compute_stream) final {
    Device device = q->device;
    TVMStreamHandle original_stream = DeviceAPI::Get(device)->GetCurrentStream(device);
    DeviceAPI::Get(device)->SetStream(device, compute_stream);
    auto [float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
          plan_info_vec] = cached_buffers_[depth];
    double rope_rcp_scale = 1 / rotary_scale;
    double rope_rcp_theta = 1 / rotary_theta;

    ICHECK_EQ(pages.ndim(), 5);
    int H = pages->shape[2];
    int N = pages->shape[3];
    int D = pages->shape[4];
    CHECK(pages.IsContiguous());
    std::vector<int64_t> pages_k_v_shape = {pages->shape[0], H, N, D};
    std::vector<int64_t> pages_k_v_strides = {2 * H * N * D, N * D, D, 1};
    Tensor pages_k =
        Tensor::FromNDAlloc(ViewBasedAlloc(pages), ffi::Shape(pages_k_v_shape), pages->dtype,
                            pages->device, pages_k_v_strides.data(), pages->byte_offset);
    Tensor pages_v = Tensor::FromNDAlloc(
        ViewBasedAlloc(pages), ffi::Shape(pages_k_v_shape), pages->dtype, pages->device,
        pages_k_v_strides.data(), pages->byte_offset + (H * N * D) * pages.DataType().bytes());

    attn_func_(float_workspace_buffer, int_workspace_buffer, plan_info_vec, q, pages_k, pages_v,
               page_indptr, page_indices, length_info, attn_output, attn_lse,
               /*layout(HND)=*/1, /*window_left=*/-1, /*enable_pdl=*/false, sm_scale,
               /*rope_rcp_scale=*/rope_rcp_scale, /*rope_rcp_theta=*/rope_rcp_theta);
    DeviceAPI::Get(device)->SetStream(device, original_stream);
  }

  void BeginForward(int depth, Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                    Tensor page_locked_int_workspace_buffer, HostMemoryVector* page_indptr,
                    int64_t batch_size, int64_t page_size, int64_t num_qo_heads,
                    int64_t num_kv_heads, int64_t qk_head_dim, int64_t v_head_dim,
                    RoPEMode rope_mode, DataType q_dtype, DataType kv_dtype,
                    TVMStreamHandle copy_stream) final {
    // Todo(tvm-team): enable cuda graph
    Tensor empty_qkv_data = Tensor::Empty({1}, q_dtype, Device{kDLCPU, 0});
    Device device = float_workspace_buffer->device;
    TVMStreamHandle original_stream = DeviceAPI::Get(device)->GetCurrentStream(device);
    DeviceAPI::Get(device)->SetStream(device, copy_stream);
    ffi::Array<int64_t> plan_info_vec =
        plan_func_(float_workspace_buffer, int_workspace_buffer, page_locked_int_workspace_buffer,
                   page_indptr->as_tensor(), batch_size, num_qo_heads, num_kv_heads, page_size,
                   /*enable_cuda_graph=*/false,
                   /*window_left=*/-1, /*logits_soft_cap=*/0.0, qk_head_dim, v_head_dim,
                   empty_qkv_data, empty_qkv_data)
            .cast<ffi::Array<int64_t>>();
    DeviceAPI::Get(device)->SetStream(device, original_stream);

    if (cached_buffers_.size() <= static_cast<size_t>(depth)) {
      cached_buffers_.resize(depth + 1);
    }
    cached_buffers_[depth] =
        std::make_tuple(float_workspace_buffer, int_workspace_buffer,
                        page_locked_int_workspace_buffer, std::move(plan_info_vec));
  }

 private:
  ffi::Function plan_func_;
  std::vector<std::tuple<Tensor, Tensor, Tensor, ffi::Array<int64_t>>> cached_buffers_;
};

/*! \brief The paged prefill with tree mask attention function base class. */
class PagedPrefillTreeMaskFunc : public AttnBackendFunc {
 public:
  explicit PagedPrefillTreeMaskFunc(ffi::Function attn_func, AttnKind attn_kind,
                                    AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(Tensor q, Tensor qo_indptr, Tensor pages, Tensor page_indptr,
                   Tensor page_indices, Tensor length_info, Tensor k_rope_pos_offset,
                   Tensor q_rope_position, Tensor tree_attn_mn_indptr, Tensor tree_attn_mask,
                   RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
                   Tensor attn_output, Tensor attn_lse, TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MHA computation is not supported by the current backend";
  }

  virtual void MLA(Tensor q, Tensor qo_indptr, Tensor pages, Tensor page_indptr,
                   Tensor page_indices, Tensor length_info, Tensor tree_attn_mn_indptr,
                   Tensor tree_attn_mask, double sm_scale, Tensor attn_output, Tensor attn_lse,
                   TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MLA computation is not supported by the current backend";
  }

  virtual void BeginForward(Tensor temp_float_attn_workspace, Tensor temp_int_attn_workspace,
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
  explicit TIRPagedPrefillTreeMaskFunc(ffi::Function attn_func, AttnKind attn_kind)
      : PagedPrefillTreeMaskFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(Tensor q, Tensor qo_indptr, Tensor pages, Tensor page_indptr, Tensor page_indices,
           Tensor length_info, Tensor k_rope_pos_offset, Tensor q_rope_position,
           Tensor tree_attn_mn_indptr, Tensor tree_attn_mask, RoPEMode rope_mode,
           double rotary_scale, double rotary_theta, double sm_scale, Tensor attn_output,
           Tensor attn_lse, TVMStreamHandle compute_stream) final {
    attn_func_(q, qo_indptr, pages, page_indptr, page_indices, length_info, k_rope_pos_offset,
               q_rope_position, attn_output, attn_lse,
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale, tree_attn_mn_indptr, tree_attn_mask);
  }
};

/*! \brief The ragged prefill with tree mask function base class. */
class RaggedPrefillTreeMaskFunc : public AttnBackendFunc {
 public:
  explicit RaggedPrefillTreeMaskFunc(ffi::Function attn_func, AttnKind attn_kind,
                                     AttnBackendKind backend_kind)
      : AttnBackendFunc(std::move(attn_func), attn_kind, backend_kind) {}

  virtual void MHA(Tensor q, Tensor k, Tensor v, Tensor qo_indptr, Tensor kv_indptr,
                   Tensor q_rope_position, Tensor tree_attn_mn_indptr, Tensor tree_attn_mask,
                   RoPEMode rope_mode, double rotary_scale, double rotary_theta, double sm_scale,
                   Tensor attn_output, Tensor attn_lse, TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MHA computation is not supported by the current backend";
  }

  virtual void MLA(Tensor q, Tensor compressed_kv, Tensor k_pe, Tensor qo_indptr, Tensor kv_indptr,
                   Tensor tree_attn_mn_indptr, Tensor tree_attn_mask, double sm_scale,
                   Tensor attn_output, Tensor attn_lse, TVMStreamHandle compute_stream) {
    LOG(FATAL) << "MLA computation is not supported by the current backend";
  }

  virtual void BeginForward(Tensor temp_float_attn_workspace, Tensor temp_int_attn_workspace,
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
  explicit TIRRaggedPrefillTreeMaskFunc(ffi::Function attn_func, AttnKind attn_kind)
      : RaggedPrefillTreeMaskFunc(std::move(attn_func), attn_kind, AttnBackendKind::kTIR) {}

  void MHA(Tensor q, Tensor k, Tensor v, Tensor qo_indptr, Tensor kv_indptr, Tensor q_rope_position,
           Tensor tree_attn_mn_indptr, Tensor tree_attn_mask, RoPEMode rope_mode,
           double rotary_scale, double rotary_theta, double sm_scale, Tensor attn_output,
           Tensor attn_lse, TVMStreamHandle compute_stream) final {
    attn_func_(q, qo_indptr, k, v, kv_indptr, q_rope_position, tree_attn_mn_indptr, tree_attn_mask,
               attn_output, attn_lse,
               /*rotary_mode=*/static_cast<int64_t>(rope_mode == RoPEMode::kInline), rotary_scale,
               rotary_theta, sm_scale);
  }
};

/*!
 * \brief Create a PagedPrefillFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention
 * ffi::Functions. \param attn_kind The attention kind of the function. \return The created
 * PagedPrefillFunc pointer.
 */
std::unique_ptr<PagedPrefillFunc> ConvertPagedPrefillFunc(ffi::Array<ffi::Any> args,
                                                          AttnKind attn_kind);

/*!
 * \brief Create a PagedDecodeFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention
 * ffi::Functions. \param attn_kind The attention kind of the function. \return The created
 * PagedDecodeFunc pointer.
 */
std::unique_ptr<PagedDecodeFunc> ConvertPagedDecodeFunc(ffi::Array<ffi::Any> args,
                                                        AttnKind attn_kind);

/*!
 * \brief Create a RaggedPrefillFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention
 * ffi::Functions. \param attn_kind The attention kind of the function. \return The created
 * RaggedPrefillFunc pointer.
 */
std::unique_ptr<RaggedPrefillFunc> ConvertRaggedPrefillFunc(ffi::Array<ffi::Any> args,
                                                            AttnKind attn_kind);

/*!
 * \brief Create a PagedPrefillTreeMaskFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention
 * ffi::Functions. \param attn_kind The attention kind of the function. \return The created
 * PagedPrefillTreeMaskFunc pointer.
 */
std::unique_ptr<PagedPrefillTreeMaskFunc> ConvertPagedPrefillTreeMaskFunc(ffi::Array<ffi::Any> args,
                                                                          AttnKind attn_kind);

/*!
 * \brief Create a RaggedPrefillTreeMaskFunc from the given arguments and the attention kind.
 * \param args The arguments that contains the backend kind and the runtime attention
 * ffi::Functions. \param attn_kind The attention kind of the function. \return The created
 * RaggedPrefillTreeMaskFunc pointer.
 */
std::unique_ptr<RaggedPrefillTreeMaskFunc> ConvertRaggedPrefillTreeMaskFunc(
    ffi::Array<ffi::Any> args, AttnKind attn_kind);

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_ATTN_BACKEND_H_
