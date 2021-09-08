def avg_pool_MxN_impl(M, N, uniq_id):
    """Emit C code for avg_pool impl."""
    cc_code = f"""
#ifndef __STATIC_FORCEINLINE
    #define __STATIC_FORCEINLINE  static inline
#endif

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t Sum16_{N}_{uniq_id}(
    int16_t *arr) {{
	int n;
	int32_t *p32;
	int32_t res;
	
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
	
	return res;
}
    """
    return cc_code
