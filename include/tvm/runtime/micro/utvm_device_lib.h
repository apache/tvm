/*!
 *  Copyright (c) 2019 by Contributors
 * \file utvm_device_lib.h
 * \brief utvm device library definitions
 */
#ifndef TVM_RUNTIME_MICRO_UTVM_DEVICE_LIB_H_
#define TVM_RUNTIME_MICRO_UTVM_DEVICE_LIB_H_

#include <stdint.h>

void* (*TVMBackendAllocWorkspace_)(int, int, uint64_t, int,
                                   int) = (void* (*)(int, int, uint64_t, int, int)) 1;
int (*TVMBackendFreeWorkspace_)(int, int, void*) = (int (*)(int, int, void*)) 1;
void (*TVMAPISetLastError_)(const char*) = (void (*)(const char*)) 1;

#ifdef __cplusplus
extern "C"
#endif
void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size,
    int dtype_code_hint, int dtype_bits_hint) {
  return (*TVMBackendAllocWorkspace_)(device_type, device_id, size, dtype_code_hint,
                                      dtype_bits_hint);
}
#ifdef __cplusplus
extern "C"
#endif
int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  return (*TVMBackendFreeWorkspace_)(device_type, device_id, ptr);
}
#ifdef __cplusplus
extern "C"
#endif
void TVMAPISetLastError(const char* msg) {
  (*TVMAPISetLastError_)(msg);
}
#ifdef __cplusplus
extern "C"
#endif
float min(float a, float b) {
  if (a < b) {
    return a;
  } else {
    return b;
  }
}
#ifdef __cplusplus
extern "C"
#endif
float max(float a, float b) {
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

#endif  // TVM_RUNTIME_MICRO_UTVM_DEVICE_LIB_H_
