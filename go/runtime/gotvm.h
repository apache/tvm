/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm native interface declaration.
 * \file gotvm.h
 *
 * These declarations are in cgo interface definition while calling API
 * across golang and native C boundaries.
 */

#ifndef GO_RUNTIME_GOTVM_H_
#define GO_RUNTIME_GOTVM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <tvm/runtime/c_runtime_api.h>
#include <dlpack/dlpack.h>

// Some type devinitions for golang "C"
typedef void *native_voidp;
typedef int64_t native_long_long;
typedef uint64_t native_u_long_long;

// golang string in native structure
typedef struct { char *p; int n; } _gostring_;

// golang TVMType
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } _tvmtype_;

// golang TVMContext
typedef struct { int device_type; int device_id; } _tvmcontext_;

// native memory allocation and free routines
extern void _native_free(void *ptr);
extern void* _native_malloc(int len);

// Version
extern _gostring_ _TVM_VERSION(void);

// Runtime C Intercafe API
extern int _TVMFuncListGlobalNames(native_voidp);
extern int _TVMFuncGetGlobal(_gostring_ funcname, native_voidp funp);
extern int _TVMModLoadFromFile(_gostring_ modpath, _gostring_ modtype, native_voidp modp);
extern int _TVMArrayAlloc(native_voidp shape, int ndim,
                          int dtype_code, int dtype_bits, int dtype_lanes,
                          int device_type, int device_id, native_voidp dltensor);
extern int _TVMArrayFree(native_voidp dltensor);
extern int _TVMModGetFunction(uintptr_t modp, _gostring_ funcname,
                              int query_imports, native_voidp funp);
extern int _TVMFuncCall(uintptr_t funp, uintptr_t arg_values,
                        native_voidp type_codes, int num_args,
                        uintptr_t ret_values, native_voidp ret_type_codes);
extern int _TVMArrayCopyFromBytes(native_voidp dltensor, native_voidp data, int size);
extern int _TVMArrayCopyToBytes(native_voidp dltensor, native_voidp data, int size);

// Error API
extern _gostring_ _TVMGetLastError(void);

// TVMValue API
extern uintptr_t _NewTVMValue();
extern void _DeleteTVMValue(uintptr_t tvmval);
extern void _TVMValueSetInt64(uintptr_t tvmval, int64_t val);
extern int64_t _TVMValueGetInt64(uintptr_t tvmval);
extern void _TVMValueSetFloat64(uintptr_t tvmval, double val);
extern double _TVMValueGetFloat64(uintptr_t tvmval);
extern void _TVMValueSetHandle(uintptr_t tvmval, uintptr_t val);
extern void * _TVMValueGetHandle(uintptr_t tvmval);
extern void _TVMValueSetStr(uintptr_t tvmval, _gostring_ val);
extern _gostring_ _TVMValueGetStr(uintptr_t tvmval);
extern void _TVMValueUnSetStr(uintptr_t tvmval);

extern native_voidp _TVMValueNativeAllocate(int len);
extern void _TVMValueNativeSet(native_voidp to, native_voidp from, int index);
extern void _TVMValueNativeGet(native_voidp to, native_voidp from, int index);
extern void _TVMValueNativeFree(native_voidp ptr);

// DLTensor API
extern uintptr_t _NewDLTensor(void);
extern void _DeleteDLTensor(uintptr_t dltensor);
extern uintptr_t _DLTensorCopyTo(uintptr_t pdltensor);
extern int _DLTensorGetNdim(uintptr_t pdltensor);
extern void * _DLTensorGetShape(uintptr_t pdltensor);
extern _tvmtype_ _DLTensorGetDType(uintptr_t pdltensor);
extern _tvmcontext_ _DLTensorGetCtx(uintptr_t pdltensor);

// TVMByteArray
extern void _TVMByteArraySetData(uintptr_t tbytearray, _gostring_ val);
extern _gostring_ _TVMByteArrayGetData(uintptr_t tbytearray);
extern void _TVMByteArraySetSize(uintptr_t tbytearray, native_long_long val);
extern native_long_long _TVMByteArrayGetSize(uintptr_t tbytearray);
extern uintptr_t _NewTVMByteArray(void);
extern void _DeleteTVMByteArray(uintptr_t tbytearray);

#ifdef __cplusplus
}
#endif
#endif  // GO_RUNTIME_GOTVM_H_
