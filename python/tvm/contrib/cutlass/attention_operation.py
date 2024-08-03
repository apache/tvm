# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Generator for CUTLASS attention kernels."""
from .library import substitute_template


def instantiate_attention_template(attrs):
    """Return CUTLASS host code for fused multi head attention
    based on a template and the provided attribute map."""

    bias_template = """
  CHECK(${bias}->ndim == 4); // B, N, S, S'

  p.attn_bias_ptr = reinterpret_cast<T *>(${bias}->data);
  p.bias_strideM = ${bias_strideM};
  p.bias_strideH = ${bias_strideH};
  p.bias_strideB = ${bias_strideB};
"""

    var_len_template = """
  p.seqstart_q_ptr = (int32_t*)${seqstart_q}->data;
  p.seqstart_k_ptr = (int32_t*)${seqstart_k}->data;
  p.num_queries = ((int32_t*)${max_seqlen_q}->data)[0];
  p.num_batches = ${seqstart_q}->shape[0] - 1;
"""

    qkv_template = {
        "default": """
  p.query_ptr = reinterpret_cast<T *>(${query}->data);
  p.key_ptr = reinterpret_cast<T *>(${key}->data);
  p.value_ptr = reinterpret_cast<T *>(${value}->data);
  CHECK(${query}->ndim == 4); // B, S, N, H
  CHECK(${key}->ndim == 4); // B, S', N, H
  CHECK(${value}->ndim == 4); // B, S', N, H'

  // stride for N
  p.q_strideH = p.head_dim; // H
  p.k_strideH = p.head_dim; // H
  p.v_strideH = p.head_dim_value; // H'

  // stride for S
  p.q_strideM = p.q_strideH * p.num_heads; // H * N
  p.k_strideM = p.k_strideH * p.num_heads; // H * N
  p.v_strideM = p.v_strideH * p.num_heads; // H' * N

  // stride for B
  p.q_strideB = p.q_strideM * p.num_queries; // H * N * S
  p.k_strideB = p.k_strideM * p.num_keys; // H * N * S'
  p.v_strideB = p.v_strideM * p.num_keys; // H'* N * S'
""",
        "qkv_stacked": """
  p.query_ptr = reinterpret_cast<T *>(${qkv}->data);
  p.key_ptr = reinterpret_cast<T *>(${qkv}->data) + p.head_dim * p.num_heads;
  p.value_ptr = reinterpret_cast<T *>(${qkv}->data) + p.head_dim * p.num_heads * 2;
  CHECK(${qkv}->ndim == 3); // B, S, NH + NH + NH'

  // stride for N
  p.q_strideH = p.head_dim; // H
  p.k_strideH = p.head_dim; // H
  p.v_strideH = p.head_dim_value; // H'

  // stride for S
  p.q_strideM = p.k_strideM = p.v_strideM =
    p.q_strideH * p.num_heads +
    p.k_strideH * p.num_heads +
    p.v_strideH * p.num_heads; // H * N + H * N + H * N'

  // stride for B
  p.q_strideB = p.k_strideB = p.v_strideB =
    p.q_strideM * p.num_queries; // (H * N + H * N + H * N') * S
""",
    }

    template = """
  using T = ${data_type};

  using Attention =
      AttentionKernel<T,
                      /*ArchTag=*/${arch},
                      /*is_aligned=*/${kIsAligned},
                      /*queries_per_block=*/${kQueriesPerBlock},
                      /*keys_per_block=*/${kKeysPerBlock},
                      /*kMaxK=*/${kMaxK},
                      /*supports_dropout=*/${kSupportsDropout},
                      /*supports_bias=*/${kSupportsBias}
      >;

  typename Attention::Params p;
  p.logsumexp_ptr = nullptr;
  p.output_ptr = reinterpret_cast<T *>(out0->data);

  p.output_accum_ptr = nullptr;
  uint64_t accumulator_buf_size = ${output_size} * sizeof(Attention::output_accum_t);
  bool accumulator_buf_allocated = false;
  if (Attention::kNeedsOutputAccumulatorBuffer) {
    if (accumulator_buf_size <= ${workspace}->shape[0]) {
        p.output_accum_ptr = static_cast<float*>(${workspace}->data);
    } else {
        accumulator_buf_allocated = true;
        cudaMalloc(
          &p.output_accum_ptr,
          accumulator_buf_size
        );
    }
  }

  p.num_heads = ${num_heads}; // N
  p.num_batches = ${num_batches}; // B
  p.head_dim = ${head_dim}; // H
  p.head_dim_value = ${head_dim_value}; // H'
  p.num_queries = ${num_queries}; // S
  p.num_keys = ${num_keys}; // S'
  p.scale = ${scale};
  p.custom_mask_type = ${custom_mask_type};


  p.o_strideM = p.head_dim_value * p.num_heads; // H' * N
  CHECK(out0->ndim == 4); // B, S, N, H'

  ${qkv_template}
  ${bias_template}
  ${var_len_template}

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    static bool once = [&]() {
      cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      return true;
    }();
  }

  CHECK(Attention::check_supported(p));
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);

  if (accumulator_buf_allocated) {
    cudaFree(p.output_accum_ptr);
  }
"""

    template = substitute_template(
        template,
        {
            "qkv_template": qkv_template[attrs["qkv_layout"]],
            "bias_template": bias_template if "bias" in attrs else "",
            "var_len_template": var_len_template if "seqstart_q" in attrs else "",
        },
    )

    return substitute_template(template, attrs)


def instantiate_flash_attention_template(attrs):
    """Return host code for flash attention."""

    template = """
    int q_head_stride = ${head_dim};
    int k_head_stride = ${head_dim};
    int v_head_stride = ${head_dim};
    int o_head_stride = ${head_dim};
    int q_row_stride = q_head_stride * ${num_q_heads};
    int k_row_stride = k_head_stride * ${num_kv_heads};
    int v_row_stride = v_head_stride * ${num_kv_heads};
    int o_row_stride = o_head_stride * ${num_q_heads};
    int q_batch_stride = q_row_stride * ${num_queries};
    int k_batch_stride = k_row_stride * ${num_keys};
    int v_batch_stride = v_row_stride * ${num_keys};
    int o_batch_stride = o_row_stride * ${num_queries};

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    flash_attn::flash_attention_forward(
                            static_cast<const cutlass::half_t*>(${query}->data),
    			    static_cast<const cutlass::half_t*>(${key}->data),
    			    static_cast<const cutlass::half_t*>(${value}->data),
    			    static_cast<cutlass::half_t*>(out0->data),
    			    ${num_batches},
    			    ${num_queries},
    			    ${num_keys},
    			    ${num_q_heads},
    			    ${num_kv_heads},
    			    ${head_dim},
    			    q_batch_stride,
    			    k_batch_stride,
    			    v_batch_stride,
    			    o_batch_stride,
    			    q_head_stride,
    			    k_head_stride,
    			    v_head_stride,
    			    o_head_stride,
    			    q_row_stride,
    			    k_row_stride,
    			    v_row_stride,
    			    o_row_stride,
    			    ${scale},
    			    ${is_causal},
                            ${window_size_left},
                            ${window_size_right},
    			    stream);
    """

    template_stacked = """
    int q_head_stride = ${head_dim};
    int k_head_stride = ${head_dim};
    int v_head_stride = ${head_dim};
    int o_head_stride = ${head_dim};
    int row_stride = q_head_stride * ${num_q_heads} +
                     k_head_stride * ${num_kv_heads} +
                     v_head_stride * ${num_kv_heads};
    int q_row_stride = row_stride;
    int k_row_stride = row_stride;
    int v_row_stride = row_stride;
    int o_row_stride = o_head_stride * ${num_q_heads};

    int q_batch_stride = q_row_stride * ${num_queries};
    int k_batch_stride = k_row_stride * ${num_keys};
    int v_batch_stride = v_row_stride * ${num_keys};
    int o_batch_stride = o_row_stride * ${num_queries};

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    flash_attn::flash_attention_forward(
                            static_cast<const cutlass::half_t*>(${qkv}->data),
    			    static_cast<const cutlass::half_t*>(${qkv}->data) + ${head_dim} * ${num_q_heads},
    			    static_cast<const cutlass::half_t*>(${qkv}->data) + ${head_dim} * (${num_q_heads} + ${num_kv_heads}),
    			    static_cast<cutlass::half_t*>(out0->data),
    			    ${num_batches},
    			    ${num_queries},
    			    ${num_keys},
    			    ${num_q_heads},
    			    ${num_kv_heads},
    			    ${head_dim},
    			    q_batch_stride,
    			    k_batch_stride,
    			    v_batch_stride,
    			    o_batch_stride,
    			    q_head_stride,
    			    k_head_stride,
    			    v_head_stride,
    			    o_head_stride,
    			    q_row_stride,
    			    k_row_stride,
    			    v_row_stride,
    			    o_row_stride,
    			    ${scale},
    			    ${is_causal},
                            ${window_size_left},
                            ${window_size_right},
    			    stream);
    """

    if "qkv" in attrs:
        return substitute_template(template_stacked, attrs)

    return substitute_template(template, attrs)


def instantiate_flash_attention_var_len_template(attrs):
    """Return host code for flash attention with variable sequence lengths."""

    template = """
    int _max_seqlen_q = ((int32_t*)${max_seqlen_q}->data)[0];
    int _max_seqlen_k = ((int32_t*)${max_seqlen_k}->data)[0];

    int batch_size = ${seqstart_q}->shape[0] - 1;

    int q_head_stride = ${head_dim};
    int k_head_stride = ${head_dim};
    int v_head_stride = ${head_dim};
    int o_head_stride = ${head_dim};
    int q_row_stride = q_head_stride * ${num_q_heads};
    int k_row_stride = k_head_stride * ${num_kv_heads};
    int v_row_stride = v_head_stride * ${num_kv_heads};
    int o_row_stride = o_head_stride * ${num_q_heads};

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    flash_attn::flash_attention_var_len_forward(
                            static_cast<const cutlass::half_t*>(${query}->data),
    			    static_cast<const cutlass::half_t*>(${key}->data),
    			    static_cast<const cutlass::half_t*>(${value}->data),
                            static_cast<const int*>(${seqstart_q}->data),
                            static_cast<const int*>(${seqstart_k}->data),
    			    static_cast<cutlass::half_t*>(out0->data),
    			    batch_size,
    			    _max_seqlen_q,
    			    _max_seqlen_k,
    			    ${num_q_heads},
    			    ${num_kv_heads},
    			    ${head_dim},
    			    q_head_stride,
    			    k_head_stride,
    			    v_head_stride,
    			    o_head_stride,
    			    q_row_stride,
    			    k_row_stride,
    			    v_row_stride,
    			    o_row_stride,
    			    ${scale},
    			    ${is_causal},
                            // For SWA, is_causal must be false.
                            ${is_causal} ? _max_seqlen_k : ${window_size_left},
                            ${window_size_right},
    			    stream);
    """

    return substitute_template(template, attrs)
