def max_pool_MxN_impl(M, N, uniq_id):
    """Emit C code for pool impl."""
    cc_code = f"""
#ifndef __STATIC_FORCEINLINE
    #define __STATIC_FORCEINLINE  static inline
#endif

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max_pool_rest(
    int N,
    int8_t *res, 
    int8_t *arg) {{
  for (int j = 0; j < N; j++)
    if ( arg[j] > res[j] )
      res[j] = arg[j];
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max_pool_{M}x{N}_loop_{uniq_id}(
    int8_t *res, 
    int8_t *arg) {{
  for (int i = 0; i < {M}; i++)
    for (int j = 0; j < {N}; j++)
			if ( arg[i * {N} + j] > res[i * {N} + j] )
				res[i * {N} + j] = arg[i * {N} + j];
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max_pool_{M}x{N}_{uniq_id}(
    int8_t *res, 
    int8_t *arg) {{
	int32_t *parg32 = (int32_t *)arg;
	int32_t *pres32 = (int32_t *)res;
	
	if ( {M} * {N} < 4 )
		return max_pool_{M}x{N}_loop_{uniq_id}(res, arg);

  for ( int i = 0; i < ({M} * {N}) / 4; ++ i ) {{
		int32_t arg32 = *parg32 ++;
		int32_t res32 = *pres32;
		__SSUB8(arg32, res32);
		res32 = __SEL(arg32, res32);
		*pres32 ++ = res32;
	}}

	if ( ({M} * {N}) % 4 != 0 )
		return max_pool_rest(({M} * {N}) % 4, (int8_t *)pres32, (int8_t *)parg32);

  return 0;
}}
    """
    return cc_code
