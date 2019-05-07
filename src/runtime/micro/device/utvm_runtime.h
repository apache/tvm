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
 * \brief task structure for uTVM
 */
typedef struct {
  uint64_t (*func)(void*, void*, int32_t);
  UTVMArgs* args;
} UTVMTask;

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
#endif  // TVM_RUNTIME_MICRO_DEVICE_UTVM_RUNTIME_H_
