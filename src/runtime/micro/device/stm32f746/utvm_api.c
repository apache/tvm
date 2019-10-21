
#ifdef __cplusplus
extern "C" {
#endif

#include "utvm_runtime.h"

#define SYST_CSR    (*((volatile unsigned long *) 0xE000E010))
#define SYST_RVR    (*((volatile unsigned long *) 0xE000E014))
#define SYST_CVR    (*((volatile unsigned long *) 0xE000E018))
#define SYST_CALIB  (*((volatile unsigned long *) 0xE000E01C))

#define SYST_CSR_ENABLE     0
#define SYST_CSR_TICKINT    1
#define SYST_CSR_CLKSOURCE  2
#define SYST_COUNTFLAG      16

#define SYST_CALIB_NOREF  31
#define SYST_CALIB_SKEW   30

unsigned long start_time = 0;
unsigned long stop_time = 0;
unsigned long duration = 0;

void UTVMTimerStart() {
    SYST_CSR = (1 << SYST_CSR_ENABLE) | (1 << SYST_CSR_CLKSOURCE);
    // wait until timer starts
    while (SYST_CVR == 0);
    start_time = SYST_CVR;
}

void UTVMTimerStop() {
    SYST_CSR = 0;
    stop_time = SYST_CVR;
}

void UTVMTimerReset() {
    SYST_CSR = 0;
    // maximum reload value (24-bit)
    SYST_RVR = (~((unsigned long) 0)) >> 8;
    SYST_CVR = 0;
}

int32_t UTVMTimerRead() {
    if (!(SYST_CSR & SYST_COUNTFLAG)) {
      return start_time - stop_time;
    } else {
      TVMAPISetLastError("timer overflowed");
      return -1;
    }
}

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
