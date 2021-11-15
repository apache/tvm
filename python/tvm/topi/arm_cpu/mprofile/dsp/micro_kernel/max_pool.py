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
"""Defines max intrinsics for elemwise max operation with v7e-m DSP instructions."""

import random
import string

import tvm
from tvm import te
from . import common


def intrin_max(shape, in_dtype, out_dtype):
    """Defines a v7e-m DSP-accelerated max pool."""
    UNIQ_ID_LEN = 8
    uniq_id = "".join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))
    func_prefix = "max8"

    assert in_dtype == "int8"
    assert out_dtype == "int8"

    x = te.placeholder(shape, name="x", dtype=in_dtype)
    k = te.reduce_axis((0, 1), name="rc")
    z = te.compute(shape, lambda *i: tvm.tir.max(x[i], axis=[k]).astype(out_dtype))

    def _intrin_func(ins, outs):
        aa = ins[0]
        cc = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    cc.dtype,
                    f"{func_prefix}_{uniq_id}",
                    aa.access_ptr("r"),
                    cc.access_ptr("w"),
                    cc.strides[0],
                )
            )
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    cc.dtype, f"{func_prefix}_reset_{uniq_id}", cc.access_ptr("w"), cc.strides[0]
                )
            )
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    binds = {
        t: tvm.tir.decl_buffer(
            t.shape,
            t.dtype,
            t.op.name,
            strides=[te.var(f"{t.op.name}_s_{i}") for i in range(0, len(t.shape))],
            offset_factor=1,
        )
        for t in [x, z]
    }

    intrin_decl = te.decl_tensor_intrin(z.op, _intrin_func, binds=binds)
    return intrin_decl, uniq_id


def max_impl(uniq_id):
    """Emit C code for pool impl."""
    cc_code = (
        common.common_includes
        + f"""


#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max8_reset_{uniq_id}(
    int8_t *res,
    int N) {{
  memset(res, (int8_t)-128, N * sizeof(*res));
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max8_loop_{uniq_id}(
    int8_t *arg,
    int8_t *res,
    int N) {{
  for ( int i = 0; i < N; ++ i )
    if ( arg[i] > res[i] )
      res[i] = arg[i];
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max8_{uniq_id}(
    int8_t *arg,
    int8_t *res,
    int N) {{
  int32_t *parg32, *pres32;
  int una_arg = (int32_t)arg & 0x3, una_res = (int32_t)res & 0x3;
  int32_t retcode = 0;

  if ( N < 4 || ((una_arg || una_res) && una_arg != una_res) ) {{
    retcode = max8_loop_{uniq_id}(arg, res, N);
    goto out;
  }}
  if ( una_arg ) {{
    int n = (4 - una_arg);
    if ( n > N || (N - n) < 4 )
      n = N;
    retcode = max8_loop_{uniq_id}(arg, res, n);
    N -= n;
    if ( N == 0 )
      goto out;
    arg += n; res += n;
  }}

  parg32 = (int32_t *)arg;
  pres32 = (int32_t *)res;

  for ( int i = 0; i < N / 4; ++ i ) {{
    int32_t arg32 = *parg32 ++;
    int32_t res32 = *pres32;
    __SSUB8(arg32, res32);
    res32 = __SEL(arg32, res32);
    *pres32 ++ = res32;
  }}

  if ( N & 0x3 ) {{
    retcode = max8_loop_{uniq_id}((int8_t *)parg32, (int8_t *)pres32, N & 0x3);
    goto out;
  }}

out:
  return retcode;
}}

"""
    )
    return cc_code
