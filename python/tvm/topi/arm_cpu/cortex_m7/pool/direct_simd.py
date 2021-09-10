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
"""Direct implementation of pool."""
import random
import string

import tvm

from tvm import te
from tvm.topi.utils import simplify, traverse_inline


def max_pool2d_direct_simd_nhwc_schedule(outs):
    """Schedule function for Cortex-M7 SIMD implementation of max_pool2d."""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "pool_max" not in op.tag:
            return

        # extract tensors
        data_vec = op.input_tensors[0]

        n, h, w, c = s[op].op.axis
        ko, ki = s[op].op.reduce_axis
        s[op].reorder(n, h, w, ko, ki, c)

        def intrin_max(in_channels):
            UNIQ_ID_LEN = 8
            uniq_id = "".join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))
            func_prefix = "max_pool8"

            if isinstance(in_channels, tvm.tir.IntImm):
                in_channels = in_channels.value

            x = te.placeholder((1, 1, 1, in_channels), name="x", dtype="int8")
            k = te.reduce_axis((0, 1), name="rc")
            z = te.compute((1, 1, 1, in_channels), lambda *i: tvm.tir.max(x[i], axis=[k]))

            def _intrin_func(ins, outs):
                aa = ins[0]
                cc = outs[0]

                def _body():
                    func_name = f"{func_prefix}_{uniq_id}"
                    ib = tvm.tir.ir_builder.create()
                    zz = tvm.tir.call_extern(cc.dtype, func_name,
                                             aa.access_ptr("r"),
                                             cc.access_ptr("w"),
                                             cc.strides[0])
                    ib.emit(zz)
                    return ib.get()

                def _reduce_reset():
                    func_name = f"{func_prefix}_reset_{uniq_id}"
                    ib = tvm.tir.ir_builder.create()
                    ib.emit(
                        tvm.tir.call_extern(
                            cc.dtype, func_name, cc.access_ptr("w"),
                            cc.strides[0]
                        )
                    )
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

            return te.decl_tensor_intrin(
                z.op, _intrin_func, binds=binds
            )

        s[op].tensorize(c, intrin_max(data_vec.shape[-1]))

    traverse_inline(s, outs[-1].op, _callback)
    return s

def max_pool_impl(uniq_id):
    """Emit C code for pool impl."""
    cc_code = f"""
#ifdef __cplusplus
extern "C"
#endif
#include <arm_math.h>
#include <arm_nnsupportfunctions.h>

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max_pool8_reset_{uniq_id}(
    int8_t *res,
    int N) {{
  memset(res, (int8_t)-128, N * sizeof(*res));  
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max_pool8_loop_{uniq_id}(
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
__STATIC_FORCEINLINE int32_t max_pool8_{uniq_id}(
    int8_t *arg, 
    int8_t *res, 
    int8 N) {{
	int32_t *parg32 = (int32_t *)arg;
	int32_t *pres32 = (int32_t *)res;
	
	if ( N < 4 )
		return max_pool8_loop_{uniq_id}(arg, res, N);

  for ( int i = 0; i < N / 4; ++ i ) {{
		int32_t arg32 = *parg32 ++;
		int32_t res32 = *pres32;
		__SSUB8(arg32, res32);
		res32 = __SEL(arg32, res32);
		*pres32 ++ = res32;
	}}

	if ( N % 4 != 0 )
		return max_pool8_loop_{uniq_id}((int8_t *)parg32, (int8_t *)pres32, N % 4);

  return 0;
}}
    """
    return cc_code
