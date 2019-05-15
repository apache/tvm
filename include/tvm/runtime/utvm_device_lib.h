#ifndef UTVM_DEVICE_LIB_H_
#define UTVM_DEVICE_LIB_H_

extern void* (*TVMBackendAllocWorkspace_)(int, int, uint64_t, int, int) = (void* (*)(int, int, uint64_t, int, int)) 1;
extern int (*TVMBackendFreeWorkspace_)(int, int, void*) = (int (*)(int, int, void*)) 1;
extern void (*TVMAPISetLastError_)(const char*) = (void (*)(const char*)) 1;

#ifdef __cplusplus
extern "C"
#endif
void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size,
    int dtype_code_hint, int dtype_bits_hint) {
  return (*TVMBackendAllocWorkspace_)(device_type, device_id, size, dtype_code_hint, dtype_bits_hint);
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

#endif  // UTVM_DEVICE_LIB_H_
