def relu_MxN_impl(M, N, uniq_id):
    """Emit C code for relu impl."""
    cc_code = f"""
#ifndef __STATIC_FORCEINLINE
    #define __STATIC_FORCEINLINE  static inline
#endif


#ifdef GROVETY_OP_BENCHMARK

#ifdef __cplusplus
extern "C"
#endif // __cplusplus
void perf_timer_start(uint32_t op_id);

#ifdef __cplusplus
extern "C"
#endif // __cplusplus
void perf_timer_stop(uint32_t op_id);

#endif // GROVETY_OP_BENCHMARK


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

#ifdef GROVETY_OP_BENCHMARK
  perf_timer_stop(3);
#endif

  return 0;
}}
    """
    return cc_code
