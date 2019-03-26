/*!
 *  Copyright (c) 2019 by Contributors
 * \file utvm_runtime.h
 * \brief utvm runtime headers
 */
#ifndef UTVM_RUNTIME_H_
#define UTVM_RUNTIME_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

/*!
 * \brief task structure for uTVM
 */
typedef struct {
  int (*func)(void*, void*, int32_t);
  void* args;
  void* arg_type_ids;
  int32_t* num_args;
} UTVMTask;

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
#endif  // UTVM_RUNTIME_H_
