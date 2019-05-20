/*!
 *  Copyright (c) 2019 by Contributors
 * \file utvm_runtime.h
 * \brief utvm runtime headers
 */
#ifndef TVM_RUNTIME_MICRO_DEVICE_UTVM_RUNTIME_H_
#define TVM_RUNTIME_MICRO_DEVICE_UTVM_RUNTIME_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <tvm/runtime/c_runtime_api.h>

/*!
 * \brief POD variant of TVMArgs
 */
typedef struct {
  TVMValue* values;
  int* type_codes;
  int32_t num_args;
} UTVMArgs;

/*!
 * \brief Task structure for uTVM
 */
typedef struct {
  void (*func)(void*, void*, int32_t);
  UTVMArgs* args;
} UTVMTask;

// TODO(weberlo): Remove duplicate docs?

/*!
 * \brief Backend function to allocate temporal workspace.
 *
 * \note The result allocate spaced is ensured to be aligned to kTempAllocaAlignment.
 *
 * \param nbytes The size of the space requested.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \param dtype_code_hint The type code of the array elements. Only used in
 * certain backends such as OpenGL.
 * \param dtype_bits_hint The type bits of the array elements. Only used in
 * certain backends such as OpenGL.
 * \return nullptr when error is thrown, a valid ptr if success
 */
void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size,
                               int dtype_code_hint, int dtype_bits_hint);

/*!
 * \brief Backend function to free temporal workspace.
 *
 * \param ptr The result allocated space pointer.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \return 0 when no error is thrown, -1 when failure happens
 *
 * \sa TVMBackendAllocWorkspace
 */
int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);

/*!
 * \brief Used for implementing C API function.
 *  Set last error message before return.
 * \param msg The error message to be set.
 */
void TVMAPISetLastError(const char* msg);

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
#endif  // TVM_RUNTIME_MICRO_DEVICE_UTVM_RUNTIME_H_
