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
#include <tvm/runtime/c_runtime_api.h>
#include <dlpack/dlpack.h>

// Some type devinitions for golang "C"
typedef void* native_voidp;

// golang TVMType
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } _tvmtype_;

// golang TVMContext
typedef struct { int device_type; int device_id; } _tvmcontext_;

// Version
extern char* _TVM_VERSION(void);

// Runtime C Intercafe API
extern int _TVMFuncListGlobalNames(void*);

// TVMValue API
extern uintptr_t _NewTVMValue();
extern void _DeleteTVMValue(uintptr_t tvmval);
extern void _TVMValueSetInt64(uintptr_t tvmval, int64_t val);
extern int64_t _TVMValueGetInt64(uintptr_t tvmval);
extern void _TVMValueSetFloat64(uintptr_t tvmval, double val);
extern double _TVMValueGetFloat64(uintptr_t tvmval);
extern void _TVMValueSetHandle(uintptr_t tvmval, uintptr_t val);
extern void * _TVMValueGetHandle(uintptr_t tvmval);
extern void _TVMValueSetStr(uintptr_t tvmval, char *val);
extern char* _TVMValueGetStr(uintptr_t tvmval);
extern void _TVMValueUnSetStr(uintptr_t tvmval);
extern void _TVMValueCopyFrom(uintptr_t tvmval, uintptr_t fromval);
extern void* _TVMValueNativeAllocate(int len);
extern void _TVMValueNativeSet(void* to, void* from, int index);
extern void _TVMValueNativeGet(void* to, void* from, int index);
extern void _TVMValueNativeFree(void* ptr);

// DLTensor API
extern uintptr_t _NewDLTensor(void);
extern void _DeleteDLTensor(uintptr_t dltensor);
extern uintptr_t _DLTensorCopyTo(uintptr_t pdltensor);
extern int _DLTensorGetNdim(uintptr_t pdltensor);
extern void * _DLTensorGetShape(uintptr_t pdltensor);
extern _tvmtype_ _DLTensorGetDType(uintptr_t pdltensor);
extern _tvmcontext_ _DLTensorGetCtx(uintptr_t pdltensor);

// TVMByteArray
extern void _TVMByteArraySetData(uintptr_t tbytearray, char* val, int len);
extern char* _TVMByteArrayGetData(uintptr_t tbytearray);
extern int _TVMByteArrayGetDataLen(uintptr_t tbytearray);
extern void _TVMByteArraySetSize(uintptr_t tbytearray, int64_t val);
extern int64_t _TVMByteArrayGetSize(uintptr_t tbytearray);
extern uintptr_t _NewTVMByteArray(void);
extern void _DeleteTVMByteArray(uintptr_t tbytearray);

// Callbacks
extern int _ConvertFunction(void* fptr, void* funp);

#ifdef __cplusplus
}
#endif
#endif  // GOTVM_GOTVM_H_
