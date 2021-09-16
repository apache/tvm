#ifndef __cortex_m7_defines_h
#define __cortex_m7_defines_h

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef   __ASM
  #define __ASM  __asm
#endif

#ifndef   __STATIC_FORCEINLINE
  #define __STATIC_FORCEINLINE  __attribute__((always_inline)) static inline
#endif


#define __PKHBT(ARG1,ARG2,ARG3) \
__extension__ \
({                          \
  uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2); \
  __ASM ("pkhbt %0, %1, %2, lsl %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  ); \
  __RES; \
 })

#define __PKHTB(ARG1,ARG2,ARG3) \
__extension__ \
({                          \
  uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2); \
  if (ARG3 == 0) \
    __ASM ("pkhtb %0, %1, %2" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2)  ); \
  else \
    __ASM ("pkhtb %0, %1, %2, asr %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  ); \
  __RES; \
 })


#ifdef __cplusplus
extern "C" {
#endif

__STATIC_FORCEINLINE uint32_t __SMLAD(uint32_t op1, uint32_t op2, uint32_t op3) {
  uint32_t result;
  __ASM volatile ("smlad %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SSUB8(uint32_t op1, uint32_t op2) {
  uint32_t result;
  __ASM volatile ("ssub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SEL(uint32_t op1, uint32_t op2) {
  uint32_t result;
  __ASM volatile ("sel %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}


__STATIC_FORCEINLINE uint32_t __SXTB16(uint32_t op1) {
  uint32_t result;
  __ASM ("sxtb16 %0, %1" : "=r" (result) : "r" (op1));
  return(result);
}

__STATIC_FORCEINLINE uint32_t __ROR(uint32_t op1, uint32_t op2) {
  op2 %= 32U;
  if (op2 == 0U)
    return op1;
  return (op1 >> op2) | (op1 << (32U - op2));
}


__STATIC_FORCEINLINE int32_t arm_nn_read_q7x4_ia(
    const int8_t **in_q7) {
  int32_t val;
  memcpy(&val, *in_q7, 4);
  *in_q7 += 4;

  return (val);
}

__STATIC_FORCEINLINE const int8_t *read_and_pad(
    const int8_t *source,
    int32_t *out1,
    int32_t *out2) {
  int32_t inA = arm_nn_read_q7x4_ia(&source);
  int32_t inAbuf1 = __SXTB16(__ROR((uint32_t)inA, 8));
  int32_t inAbuf2 = __SXTB16(inA);

#ifndef ARM_MATH_BIG_ENDIAN
  *out2 = (int32_t)(__PKHTB(inAbuf1, inAbuf2, 16));
  *out1 = (int32_t)(__PKHBT(inAbuf2, inAbuf1, 16));
#else
  *out1 = (int32_t)(__PKHTB(inAbuf1, inAbuf2, 16));
  *out2 = (int32_t)(__PKHBT(inAbuf2, inAbuf1, 16));
#endif

  return source;
}

#ifdef __cplusplus
}
#endif

#endif
