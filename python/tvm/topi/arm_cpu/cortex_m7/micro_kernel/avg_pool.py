def avg_pool_MxN_impl(M, N, uniq_id):
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

#ifdef __cplusplus
extern "C"
#endif // __cplusplus
void perf_timer_stop(uint32_t op_id);

#endif // GROVETY_OP_BENCHMARK

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t Sum16_{N}_{uniq_id}(
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
