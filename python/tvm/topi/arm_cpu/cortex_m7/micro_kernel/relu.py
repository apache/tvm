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
"""Defines relu intrinsics for SIMD relu operation."""

def relu_MxN_impl(M, N, uniq_id):
    """Emit C code for relu impl."""
    cc_code = f"""
#ifndef __STATIC_FORCEINLINE
    #define __STATIC_FORCEINLINE  static inline
#endif

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t relu_rest(
    int N,
    int8_t *mat) {{
  for (int j = 0; j < N; j++)
    mat[j] = mat[j] > 0 ? mat[j] : 0;
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t relu_{M}x{N}_loop_{uniq_id}(
    int8_t *mat) {{
  for (int i = 0; i < {M}; i++)
    for (int j = 0; j < {N}; j++)
			mat[i * {N} + j] > 0 ? mat[i * {N} + j] : 0;
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t relu_{M}x{N}_{uniq_id}(
    int8_t *mat) {{

	int32_t *pmat32 = (int32_t *)mat;

#ifdef GROVETY_OP_BENCHMARK
  perf_timer_start(3);
#endif

	if ( {M} * {N} < 4 )
		return relu_{M}x{N}_loop_{uniq_id}(mat);

  for ( int i = 0; i < ({M} * {N}) / 4; ++ i ) {{
		__SSUB8(*pmat32, 0);
		*pmat32 = __SEL(*pmat32, 0);
		++ pmat32;
	}}

	if ( ({M} * {N}) % 4 != 0 )
		return relu_rest(({M} * {N}) % 4, (int8_t *)pmat32);

  return 0;
}}
    """
    return cc_code
