
#ifdef __cplusplus
extern "C" {
#endif

#include "utvm_runtime.h"

// TODO(weberlo): use this? https://stackoverflow.com/questions/5141960/get-the-current-time-in-c

void UTVMInit() { }

void UTVMTimerStart() { }

void UTVMTimerStop() { }

void UTVMTimerReset() { }

int32_t UTVMTimerRead() {
    return 420;
}

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
