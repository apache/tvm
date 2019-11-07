
#ifdef __cplusplus
extern "C" {
#endif

#include "utvm_runtime.h"

// TODO(weberlo): use this? https://stackoverflow.com/questions/5141960/get-the-current-time-in-c

void UTVMInit() {
  UTVMMain();
}

int32_t UTVMTimerStart() {
  return 0;
}

void UTVMTimerStop() { }

void UTVMTimerReset() { }

uint32_t UTVMTimerRead() {
  return 1;
}

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
