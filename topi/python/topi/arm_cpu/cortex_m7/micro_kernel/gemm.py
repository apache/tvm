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
# pylint: disable=invalid-name, no-value-for-parameter
"""Defines gemm intrinsics for SIMD matrix multiplication."""

import random
import string

import tvm
from tvm import te

##########################
# MxKxN MatMul Intrinsic #
##########################

# NOTE this is transposed matmul (A * B^T)
def intrin_gemm_MxKxN(M, K, N, in_dtype, out_dtype):
    """Defines a SIMD-accelerated transposed matmul."""
    # we generate a unique ID for every intrinsic definition, to prevent name
    # collisions in the generated source (e.g., if there are multiple operators
    # in the same module that use the same intrinsic)
    #
    # TODO(weberlo, areusch): to cut down on memory usage, we should cache each intrinsic
    # instantiation and include it only once, eliminating the need for unique
    # IDs
    UNIQ_ID_LEN = 8
    uniq_id = ''.join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))

    if isinstance(M, tvm.tir.IntImm):
        M = M.value
    if isinstance(K, tvm.tir.IntImm):
        K = K.value
    if isinstance(N, tvm.tir.IntImm):
        N = N.value
    assert K % 4 == 0
    # TODO(weberlo, areusch): support more dtypes?
    assert in_dtype == 'int8'
    assert out_dtype == 'int32'
    A = te.placeholder((M, K), name='a', dtype=in_dtype)
    B = te.placeholder((N, K), name='b', dtype=in_dtype)
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k].astype(out_dtype) * B[j, k].astype(out_dtype), axis=k),
        name='c')
    A_buf = tvm.tir.decl_buffer(
        A.shape, A.dtype,
        name="A",
        offset_factor=1,
        strides=[te.var("A_s"), 1])
    B_buf = tvm.tir.decl_buffer(
        B.shape, B.dtype,
        name="B",
        offset_factor=1,
        strides=[te.var("B_s"), 1])
    C_buf = tvm.tir.decl_buffer(
        C.shape, C.dtype,
        name="C",
        offset_factor=1,
        strides=[te.var("C_s"), 1])
    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]
        def _reduce_update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_extern("int32", f"gemm_{M}x{K}x{N}_update_{uniq_id}",
                                        aa.access_ptr("r"),
                                        bb.access_ptr("r"),
                                        cc.access_ptr("w"),
                                        aa.strides[0],
                                        bb.strides[0],
                                        cc.strides[0]))
            return ib.get()
        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_extern("int32", f"gemm_{M}x{K}x{N}_reset_{uniq_id}",
                                        cc.access_ptr("w"),
                                        cc.strides[0]))
            return ib.get()
        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_extern("int32", f"gemm_{M}x{K}x{N}_body_{uniq_id}",
                                        aa.access_ptr("r"),
                                        bb.access_ptr("r"),
                                        cc.access_ptr("w"),
                                        aa.strides[0],
                                        bb.strides[0],
                                        cc.strides[0]))
            return ib.get()
        return _body(), _reduce_reset(), _reduce_update()
    with tvm.target.build_config(offset_factor=1):
        intrin_decl = te.decl_tensor_intrin(
            C.op, intrin_func, binds={A: A_buf, B: B_buf, C: C_buf})
        return intrin_decl, uniq_id


def gemm_MxKxN_impl(M, K, N, uniq_id):
    """Emit C code for gemm impl."""
    # TODO(weberlo, areusch): are there any SIMD tricks to zero out arrays quickly?
    aa_pad_size = M * K
    bb_pad_size = N * K
    # code reference: CMSIS-NN paper (https://arxiv.org/abs/1801.06601)
    cc_code = f"""
#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_{M}x{K}x{N}_body_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  int16_t aa_pad[{aa_pad_size}];
  int16_t bb_pad[{bb_pad_size}];

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {K} / 4; j++) {{
      read_and_pad(&aa[i*A_stride + j*4], (int32_t*) &aa_pad[i*{K} + j*4], (int32_t*) &aa_pad[i*{K} + j*4 + 2]);
    }}
  }}

  for (int i = 0; i < {N}; i++) {{
    for (int j = 0; j < {K} / 4; j++) {{
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*{K} + j*4], (int32_t*) &bb_pad[i*{K} + j*4 + 2]);
    }}
  }}

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      int32_t sum = 0;
      for (int l = 0; l < {K} / 2; l++) {{
        sum = __SMLAD(
          *((int32_t*) &aa_pad[i*{K} + l*2]),
          *((int32_t*) &bb_pad[j*{K} + l*2]),
          sum);
      }}
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }}
  }}

  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_{M}x{K}x{N}_update_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  int16_t aa_pad[{aa_pad_size}];
  int16_t bb_pad[{bb_pad_size}];

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {K} / 4; j++) {{
      read_and_pad(&aa[i*A_stride + j*4], (int32_t*) &aa_pad[i*{K} + j*4], (int32_t*) &aa_pad[i*{K} + j*4 + 2]);
    }}
  }}

  for (int i = 0; i < {N}; i++) {{
    for (int j = 0; j < {K} / 4; j++) {{
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*{K} + j*4], (int32_t*) &bb_pad[i*{K} + j*4 + 2]);
    }}
  }}

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      int32_t sum = 0;
      for (int l = 0; l < {K} / 2; l++) {{
        sum = __SMLAD(
          *((int32_t*) &aa_pad[i*{K} + l*2]),
          *((int32_t*) &bb_pad[j*{K} + l*2]),
          sum);
      }}
      cc[i*C_stride + j] += sum;
    }}
  }}

  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_{M}x{K}x{N}_reset_{uniq_id}(int32_t *cc, int C_stride) {{
  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      cc[i*C_stride + j] = 0;
    }}
  }}
  return 0;
}}
    """
    return cc_code
