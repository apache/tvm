cc_code = """

#include <stdint.h>
#include <stdlib.h>
#include <string.h>


#ifndef   __ASM
  #define __ASM  __asm
#endif


#ifndef   __STATIC_FORCEINLINE
  #define __STATIC_FORCEINLINE  __attribute__((always_inline)) static inline
#endif


#ifndef __SSUB8
#define __SSUB8(OP1, OP2)                                                       \\
__extension__                                                                   \\
({                                                                              \\
  uint32_t __RES;                                                               \\
  __ASM volatile ("ssub8 %0, %1, %2" : "=r" (__RES) : "r" (OP1), "r" (OP2) );   \\
  __RES;                                                                        \\
 })
#endif // __SSUB8


#ifndef __SEL
#define __SEL(OP1, OP2)                                                         \\
__extension__                                                                   \\
({                                                                              \\
  uint32_t __RES;                                                               \\
  __ASM volatile ("sel %0, %1, %2" : "=r" (__RES) : "r" (OP1), "r" (OP2) );     \\
  __RES;                                                                        \\
 })
#endif // __SEL


#ifndef __SXTB16
#define __SXTB16(OP1)                                                           \\
__extension__                                                                   \\
({                                                                              \\
  uint32_t __RES;                                                               \\
  __ASM volatile ("sxtb16 %0, %1" : "=r" (__RES) : "r" (OP1) );                 \\
  __RES;                                                                        \\
 })
#endif // __SXTB16


#ifndef __ROR
#define __ROR(OP1, OP2)                                                         \\
__extension__                                                                   \\
({                                                                              \\
  OP2 %= 32U;                                                                   \\
  OP2 == 0U ? OP1 : (OP1 >> OP2) | (OP1 << (32U - OP2));                        \\
 })
#endif // __ROR


#ifndef __SMLAD
#define __SMLAD(OP1, OP2, OP3)                                                  \\
__extension__                                                                   \\
({                                                                              \\
  uint32_t __RES;                                                               \\
  __ASM volatile ("smlad %0, %1, %2, %3" : "=r" (__RES) : "r" (OP1), "r" (OP2), "r" (OP3) );  \\
  __RES;                                                                        \\
 })
#endif // __SMLAD


"""
