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
# pylint: disable=invalid-name, unused-wildcard-import, wildcard-import
"""Generator for CUTLASS attention kernels."""
from .library import *


def instantiate_attention_template(attrs, func_args):
    """Return CUTLASS host code for fused multi head attention
    based on a template and the provided attribute map."""

    bias_template = {
        "B11S'": """
  CHECK(${arg3}->ndim == 2); // B, 1, 1, S'

  p.attn_bias_ptr = reinterpret_cast<T *>(${arg3}->data);
  p.bias_strideM = 0; // 0
  p.bias_strideH = 0; // 0
  p.bias_strideB = p.num_keys; // S'
""",
        "B1SS'": """
  CHECK(${arg3}->ndim == 3); // B, 1, S, S'

  p.attn_bias_ptr = reinterpret_cast<T *>(${arg3}->data);
  p.bias_strideM = p.num_keys; // S'
  p.bias_strideH = 0; // 0
  p.bias_strideB = p.bias_strideM * p.num_queries; // S' * S
""",
        "BNSS'": """
  CHECK(${arg3}->ndim == 4); // B, N, S, S'

  p.attn_bias_ptr = reinterpret_cast<T *>(${arg3}->data);
  p.bias_strideM = p.num_keys; // S'
  p.bias_strideH = p.bias_strideM * p.num_queries; // S' * S
  p.bias_strideB = p.bias_strideH * p.num_heads; // S' * S * N
""",
    }

    template = """
  using T = ${data_type};

  CHECK(${arg0}->ndim == 4); // B, S, N, H
  CHECK(${arg1}->ndim == 4); // B, S', N, H
  CHECK(${arg2}->ndim == 4); // B, S', N, H'
  CHECK(out0->ndim == 4); // B, S, N, H'

  using Attention =
      AttentionKernel<T,
                      /*ArchTag=*/${arch},
                      /*is_aligned=*/${kIsAligned},
                      /*queries_per_block=*/${kQueriesPerBlock},
                      /*keys_per_block=*/${kKeysPerBlock},
                      /*single_value_iteration=*/${kSingleValueIteration},
                      /*supports_dropout=*/${kSupportsDropout},
                      /*supports_bias=*/${kSupportsBias}
      >;

  typename Attention::Params p;

  p.query_ptr = reinterpret_cast<T *>(${arg0}->data);
  p.key_ptr = reinterpret_cast<T *>(${arg1}->data);
  p.value_ptr = reinterpret_cast<T *>(${arg2}->data);
  p.logsumexp_ptr = nullptr;
  p.output_ptr = reinterpret_cast<T *>(out0->data);
  p.output_accum_ptr = nullptr;
  if (Attention::kNeedsOutputAccumulatorBuffer) {
    cudaMalloc(
      &p.output_accum_ptr,
      ${output_size} * sizeof(Attention::output_accum_t)
    );
  }

  p.num_heads = ${num_heads}; // N
  p.num_batches = ${num_batches}; // B
  p.head_dim = ${head_dim}; // H
  p.head_dim_value = ${head_dim_value}; // H'
  p.num_queries = ${num_queries}; // S
  p.num_keys = ${num_keys}; // S'
  p.scale = 1.0f / sqrt(float(${head_dim}));

  // stride for N
  p.q_strideH = p.head_dim; // H
  p.k_strideH = p.head_dim; // H
  p.v_strideH = p.head_dim_value; // H'

  // stride for S
  p.q_strideM = p.q_strideH * p.num_heads; // H * N
  p.k_strideM = p.k_strideH * p.num_heads; // H * N
  p.v_strideM = p.v_strideH * p.num_heads; // H' * N
  p.o_strideM = p.head_dim_value * p.num_heads; // H' * N

  // stride for B
  p.q_strideB = p.q_strideM * p.num_queries; // H * N * S
  p.k_strideB = p.k_strideM * p.num_keys; // H * N * S'
  p.v_strideB = p.v_strideM * p.num_keys; // H'* N * S'

  ${bias_template}

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
  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
"""
    if attrs["kSupportsBias"]:
        template = substitute_template(
            template, {"bias_template": bias_template[attrs["bias_layout"]]}
        )
    else:
        template = substitute_template(template, {"bias_template": ""})
    for i, arg in enumerate(func_args):
        attrs["arg{}".format(i)] = arg
    return substitute_template(template, attrs)
