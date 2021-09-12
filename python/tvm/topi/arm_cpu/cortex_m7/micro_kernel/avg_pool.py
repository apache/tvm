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


def intrin_sum(in_channels, in_dtype, out_dtype):
    UNIQ_ID_LEN = 8
    uniq_id = "".join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))
    func_prefix = "sum16"

    if isinstance(in_channels, tvm.tir.IntImm):
        in_channels = in_channels.value

    assert in_dtype == "int16"
    assert out_dtype == "int16"

    x = te.placeholder((1, 1, 1, in_channels), name="x", dtype=in_dtype)
    k = te.reduce_axis((0, 1), name="rc")
    z = te.compute((1, 1, 1, in_channels), lambda *i: tvm.tir.sum(x[i], axis=[k]).astype(out_dtype))

    def _intrin_func(ins, outs):
        aa = ins[0]
        cc = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(cc.dtype,
                                    f"{func_prefix}_{in_channels}_{uniq_id}",
                                    aa.access_ptr("r"),
                                    cc.access_ptr("w")))
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_extern(cc.dtype,
                                        f"{func_prefix}_{in_channels}_reset_{uniq_id}",
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
    """Emit C code for avg_pool impl."""
    cc_code = f"""
#ifndef __STATIC_FORCEINLINE
  #define __STATIC_FORCEINLINE  static inline
#endif

#ifdef GROVETY_OP_BENCHMARK

#ifdef __cplusplus
extern "C"
#endif // __cplusplus
void perf_timer_start(uint32_t op_id);
#endif
__STATIC_FORCEINLINE int32_t sum16_{N}_reset_{uniq_id}(
    int16_t *res) {{
  memset(res, (int16_t)0, {N} * sizeof(*res));  
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif // __cplusplus
void perf_timer_stop(uint32_t op_id);

#endif // GROVETY_OP_BENCHMARK

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t sum16_{N}_{uniq_id}(
    int16_t *arr) {{
  int n;
  int32_t *p32;
  int32_t res;

#ifdef GROVETY_OP_BENCHMARK
  perf_timer_start(2);
#endif
  
  if ( (long)arr % 4 != 0 ) {{
    res = *arr;
    p32 = (int32_t *)(&arr[1]);
    n = {N} - 1;
  }} else {{
    res = 0;
    p32 = (int32_t *)arr;
    n = {N};
  }}
  
  for ( int i = 0; i < n / 2; ++ i ) {{
    res += __SMUAD(*p32, 0x00010001);
    ++ p32;
  }}
  
  if ( n % 2 != 0 ) 
    res += *(int16_t *)p32;

#ifdef GROVETY_OP_BENCHMARK
  perf_timer_stop(2);
#endif

  return res;
}}
"""
    return cc_code
