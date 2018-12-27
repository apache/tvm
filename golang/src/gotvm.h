/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm native interface declaration.
 * \file gotvm.h
 *
 * These declarations are in cgo interface definition while calling API
 * across golang and native C boundaries.
 */

#ifndef GOTVM_GOTVM_H_
#define GOTVM_GOTVM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <dlpack/dlpack.h>

// Some type definitions for golang "C"
typedef void* native_voidp;

// Version
extern char* _TVM_VERSION(void);

// Wrappers : For incompatible cgo API.
// To handle array of strings wrapped into __gostring__
extern int _TVMFuncListGlobalNames(void*);
// To handle TVMValue slice to/from native sequential TVMValue array.
extern void _TVMValueNativeSet(void* to, void* from, int index);
extern void _TVMValueNativeGet(void* to, void* from, int index);

// Callbacks
extern int _ConvertFunction(void* fptr, void* funp);

#ifdef __cplusplus
}
#endif
#endif  // GOTVM_GOTVM_H_
