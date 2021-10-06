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
"""Defines max intrinsics for SIMD max operation."""

import random
import string

import tvm
from tvm import te


def intrin_sum(shape, in_dtype, out_dtype):
    UNIQ_ID_LEN = 8
    uniq_id = "".join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))
    func_prefix = "sum16"

    assert in_dtype == "int16"
    assert out_dtype == "int16"

    width = shape[-1]
    x = te.placeholder(shape, name="x", dtype=in_dtype)
    k = te.reduce_axis((0, width), name="rc")

    def get_slice(indices, k):
      slice = list(indices)
      slice[-1] = slice[-1] + k
      return tuple(slice)
    z = te.compute((1,) * len(shape), lambda *i: te.sum(x[get_slice(i, k)], axis=[k]).astype(out_dtype))

    def _intrin_func(ins, outs):
        aa = ins[0]
        cc = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(cc.dtype,
                                    f"{func_prefix}_{width}_{uniq_id}",
                                    aa.access_ptr("r"),
                                    cc.access_ptr("w"),
                                    aa.elem_offset))
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_extern(cc.dtype,
                                        f"{func_prefix}_{width}_reset_{uniq_id}",
                                        cc.access_ptr("w")))
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    binds = {
        t: tvm.tir.decl_buffer(
            t.shape, t.dtype, t.op.name,
            strides=[te.var(f"{t.op.name}_s_{i}") for i in range(0, len(t.shape))],
            offset_factor=1
        )
        for t in [x, z]
    }

    intrin_decl = te.decl_tensor_intrin(z.op, _intrin_func, binds=binds)
    return intrin_decl, uniq_id


def sum_impl(N, uniq_id):
    """Emit C code for sum impl."""
    cc_code = f"""

#ifndef   __STATIC_FORCEINLINE
  #define __STATIC_FORCEINLINE  __attribute__((always_inline)) static inline
#endif

#ifdef __cplusplus
extern "C"
#endif // __cplusplus
__STATIC_FORCEINLINE int32_t sum16_{N}_reset_{uniq_id}(
    int16_t *res) {{
  memset(res, (int16_t)0, {N} * sizeof(*res));
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t sum16_{N}_{uniq_id}(
    int16_t *arr,
    int16_t *res16,
    long arr_offset) {{
  int n;
  int32_t *p32;
  int32_t res;

#ifdef GROVETY_OP_BENCHMARK
  perf_timer_start(2);
#endif

  if ( arr_offset % 4 != 0 ) {{
    res = *arr;
    p32 = (int32_t *)(&arr[1]);
    n = {N} - 1;
  }} else {{
    res = 0;
    p32 = (int32_t *)arr;
    n = {N};
  }}

  for ( int i = 0; i < n / 2; ++ i ) {{
    res = __SMLAD(*p32, 0x00010001, res);
    ++ p32;
  }}

  if ( n % 2 != 0 )
    res += *(int16_t *)p32;

  *res16 = res;

  return 0;
}}
"""
    return cc_code
