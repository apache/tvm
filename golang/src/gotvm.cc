/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm native interface definition
 * \file gotvm.cxx
 */

// Standard includes
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <stdint.h>

// golang string compatible definition
typedef struct { char *p; int n; } _gostring_;
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

// TVM runtime C interface
#include <tvm/runtime/c_runtime_api.h>
#include <dlpack/dlpack.h>

/*!
 * \brief Memory allocation native helper
 *
 * \param len required memory size.
 *
 * \return pointer to dynamically allocated memory.
 */
void * _native_malloc(int len) {
  return malloc(len);
}

/*!
 * \brief Memory free native helper
 *
 * \param ptr is a pointer to the memory to be freed.
 */
void _native_free(void *ptr) {
  free(ptr);
}

/*!
 * \brief Convert native char array to _gostring_ structure.
 * _gostring_ structure represents the same memory footprint as golang string object.
 *
 * \param p is char pointer to a char array.
 * \param l is the size of the char array. this method exclusively need length as
 * its possible to have a bytearray in a string.
 *
 * \return _gostring_ object corresponding to native char array.
 * Caller is responsible to free the memory block allocated here.
 */
static _gostring_ _native_to_gostring(const char *p, size_t l) {
  _gostring_ ret;
  ret.p = reinterpret_cast<char*>(_native_malloc(l));
  memcpy(ret.p, p, l);
  ret.n = l;
  return ret;
}

/*!
 * \brief embeds a 64bit uint value inside a string to serialize the data.
 *
 * \param s is string object.
 * \param off is the offset in the string object.
 * \param v is the uint64_t value which need to embed into given string.
 */
static void putuint64(std::string *s, size_t off, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        (*s)[off + i] = (v >> (i * 8)) & 0xff;
    }
}

// TVM runtime C interface wrappers

/*!
 * \brief Native interface to query TVM_VERSION in golang string format.
 *
 * \return char pointer to TVM-VERSION
 */
const char* _TVM_VERSION(void) {
  const char *version = TVM_VERSION;
  return version;
}

/*!
 * \brief Native interface for getting TVMGlobal function list.
 *
 * \param names return by argument to return the function names.
 * We wrap all strings into single string joined by (len+string)
 * which is unpacked and processed in golang.
 *
 * \return c_runtime_api return status.
 */
int _TVMFuncListGlobalNames(_gostring_* names) {
  int names_size;
  char **names_array;
  int result;

  result = TVMFuncListGlobalNames(&names_size, (char const ***)&names_array);

  size_t tot = 8;
  for (int ii = 0; ii < names_size ; ++ii) {
    tot += 8 + strlen(names_array[ii]);
  }

  std::string str;
  str.resize(tot);
  putuint64(&str, 0, names_size);
  size_t off = 8;
  for (int64_t ii = 0; ii < names_size ; ++ii) {
    putuint64(&str, off, strlen(names_array[ii]));
    off += 8;
    str.replace(off, strlen(names_array[ii]), names_array[ii]);
    off += strlen(names_array[ii]);
  }
  *names = _native_to_gostring(str.data(), str.size());
  return result;
}

/*!
 * \brief Native interface for TVMGetLastError
 *
 * \return The error string if any.
 */
const char* _TVMGetLastError(void) {
  return TVMGetLastError();
}

/*!
 * \brief Native interface for TVMAPISetLastError
 *
 * \param The error string.
 */
void _TVMAPISetLastError(char *err) {
  TVMAPISetLastError(err);
  _native_free(err);
  return;
}

/*!
 * \brief Native interface for TVMModLoadFromFile
 *
 * \param modpath is compiled module path.
 * \param modtype is module type in ["so", "dso", "dll", ..etc]
 * \param modp is pointer to TVMModuleHandle is a return argument.
 *
 * \return returns status of c runtime api.
 */
int _TVMModLoadFromFile(char *modpath, char *modtype, TVMModuleHandle *modp) {
  if (!modpath) {
    if (modtype) _native_free(modtype);
    return -1;
  }
  if (!modtype) {
    if (modpath) _native_free(modpath);
    return -1;
  }

  int ret =  TVMModLoadFromFile((char const *)modpath, (char const *)modtype, modp);

  _native_free(modpath);
  _native_free(modtype);
  return ret;
}

/*!
 * \brief Native interface for TVMFuncGetGlobal
 *
 * \param funcname is char pointer to function name
 * \param funp is pointer to TVMModuleHandle which is return value by argument.
 *
 * \return native api resutn status.
 */
int _TVMFuncGetGlobal(char *funcname, TVMModuleHandle *funp) {
  if (!funcname) {
    return -1;
  }

  int ret =  TVMFuncGetGlobal((char const *)funcname, funp);

  _native_free(funcname);
  return ret;
}

/*!
 * \brief Native interface for TVMArrayAlloc
 *
 * \param shape is pointer to an array of long long.
 * \param ndim indicates size of shape array.
 * \param dtype_code is the type code in DLDataType or TVMType.
 * \param dtype_bits is the bits in DLDataType or TVMType.
 * \param dtype_lanes is the lanes in DLDataType or TVMType.
 * \param device_type is the target device type in TVMContext.
 * \param device_is is the device id of given target in TVMContext.
 *
 * \return is the status of c runtime api call.
 */
int _TVMArrayAlloc(int64_t *shape, int ndim,
                   int dtype_code, int dtype_bits, int dtype_lanes,
                   int device_type, int device_id, TVMArrayHandle *dltensor) {
  return TVMArrayAlloc(reinterpret_cast<tvm_index_t *>(shape), ndim,
                       dtype_code, dtype_bits, dtype_lanes,
                       device_type, device_id, dltensor);
}

/*!
 * \brief Native interface for TVMArrayFree
 *
 * \param dltensor is pointer to a valis dltensor which need to be freed.
 *
 * \return the status of c runtime api call.
 */
int _TVMArrayFree(DLTensor *dltensor) {
  return TVMArrayFree(dltensor);
}

/*!
 * \brief Native interface for TVMModGetFunction
 *
 * \param modp is TVMModuleHandle
 * \param funcname contain the function name in given module.
 * \param query_imports indicates flag to query the dependent imported modules or not.
 * \param funp is return value by argument for the packed function from c runtime api.
 *
 * \return the status of c runtime api call.
 */
int _TVMModGetFunction(TVMModuleHandle modp, char *funcname,
                       int query_imports, TVMFunctionHandle *funp) {
  if (!funcname) {
    return -1;
  }

  int result = TVMModGetFunction(modp, (char const *)funcname, query_imports, funp);

  _native_free(funcname);
  return result;
}

/*!
 * \brief Native interface for TVMFuncCall
 *
 * \param funp is the TVMFunctionHandle for packed function.
 * \param arg_values is pointer to an array of TVMValue arguments ot the function.
 * \param type_codes is pointer to an array of TVMType corresponding to arg_values.
 * \param num_args indicates number of arguments to the function.
 * \param ret_values is pointer to TVMValue which is return argument.
 * \param ret_type_codes is pointer to TVMType dode which is return argument
 *
 * \return the status of c runtime api call.
 */
int _TVMFuncCall(TVMFunctionHandle funp, TVMValue *arg_values, int *type_codes,
                 int num_args, TVMValue *ret_values, int *ret_type_codes) {
  return TVMFuncCall(funp, arg_values, type_codes, num_args, ret_values, ret_type_codes);
}

/*!
 * \brief Native interface for TVMArrayCopyFromBytes
 *
 * \param dltensor if a pointer to TVMArray.
 * \param data is pointer to bytes.
 * \param size is the length of data to be copied.
 *
 * \return the status of C runtime api call
 */
int _TVMArrayCopyFromBytes(DLTensor *dltensor, void *data, int64_t size) {
    return TVMArrayCopyFromBytes(dltensor, data, size);
}

/*!
 * \brief Native interface for TVMArrayCopyToBytes
 *
 * \param dltensor if a pointer to TVMArray.
 * \param data is pointer to bytes.
 * \param size is the length of data to be copied back.
 *
 * \return the status of C runtime api call
 */
int _TVMArrayCopyToBytes(DLTensor *dltensor, void *data, int64_t size) {
    return TVMArrayCopyToBytes(dltensor, data, size);
}

/*!
 * \brief Native interface for TVMCFuncSetReturn
 *
 * \param ret_handle is TVMRetValueHandle for this call.
 * \param ret_value is pointer to TVMValue
 * \param type_code is the return type code for TVMValue.
 * \param num_args indicates number of return values. Only support 1 now.
 *
 * \return the status of C runtime api call
 */
int _TVMCFuncSetReturn(TVMRetValueHandle ret_handle,
                       TVMValue *ret_value,
                       int *type_code,
                       int num_args) {
    return TVMCFuncSetReturn(ret_handle, ret_value, type_code, num_args);
}

// Helpers for TVMValue

/*!
 * \brief Native helper to TVMValue creation
 *
 * \return a new TVMValue
 */
TVMValue * _NewTVMValue() {
  return reinterpret_cast<TVMValue *>(new TVMValue());
}

/*!
 * \brief Native helper to TVMValue free
 *
 * \param tvmval pointer to a valid TVMValue
 */

void _DeleteTVMValue(TVMValue *tvmval) {
  delete tvmval;
}

/*!
 * \brief Native helper to copy TVMValue object
 *
 * \param tvmval pointer to a valid TVMValue
 * \param fromval is pointer to valid TVMValue
 */
void _TVMValueCopyFrom(TVMValue *tvmval, TVMValue *fromval) {
    memcpy(tvmval, fromval, sizeof(TVMValue));
}

/*!
 * \brief Native helper to int64 setter
 *
 * \param tvmval is a valid TVMValue pointer.
 * \param val is long long aka int64 to set the value.
 */
void _TVMValueSetInt64(TVMValue *tvmval, uint64_t val) {
  tvmval->v_int64 = val;
}

/*!
 * \brief Native helper to int64 getter
 *
 * \param tvmval is a valid TVMValue Pointer.
 *
 * \return v_int64 inside TVMValue.
 */
int64_t _TVMValueGetInt64(TVMValue *tvmval) {
  return tvmval->v_int64;
}

/*!
 * \brief Native helper to float64 setter
 *
 * \param tvmval is a valid TVMValue pointer.
 * \param val is double aka float64 to set the value.
 */
void _TVMValueSetFloat64(TVMValue *tvmval, double val) {
  tvmval->v_float64 = val;
}

/*!
 * \brief Native helper to float64 getter
 *
 * \param tvmval is a valid TVMValue Pointer.
 *
 * \return v_float64 inside TVMValue.
 */
double _TVMValueGetFloat64(TVMValue *tvmval) {
  return tvmval->v_float64;
}

/*!
 * \brief Native helper to v_handle setter
 *
 * \param tvmval is a valid TVMValue pointer.
 * \param val is the void pointer handle to set the value.
 */
void _TVMValueSetHandle(TVMValue *tvmval, void* val) {
  tvmval->v_handle = val;
}

/*!
 * \brief Native helper to handle getter
 *
 * \param tvmval is a valid TVMValue Pointer.
 *
 * \return v_handle inside TVMValue.
 */
void * _TVMValueGetHandle(TVMValue *tvmval) {
  return tvmval->v_handle;
}

/*!
 * \brief Native helper to v_str setter
 *
 * \param tvmval is a valid TVMValue pointer.
 * \param val is char pointer to string.
 */
void _TVMValueSetStr(TVMValue *tvmval, char *val) {
  tvmval->v_str = val;
}

/*!
 * \brief Native helper to handle getter
 *
 * \param tvmval is a valid TVMValue Pointer.
 *
 * \return char array v_str inside TVMValue.
 */
const char* _TVMValueGetStr(TVMValue *tvmval) {
  return tvmval->v_str;
}

/*!
 * \brief Native helper to free the v_str memory in TVMValue.
 * This method used to free the memory allocated in _TVMValueSetStr.
 *
 * \param tvmval is a valid TVMValue Pointer.
 */
void _TVMValueUnSetStr(TVMValue *tvmval) {
  if (tvmval->v_str) {
    delete (tvmval->v_str);
    tvmval->v_str = 0;
  }
}

/*!
 * \brief Native helper to allocate memory for TVMValue array.
 *
 * \param n is the array length.
 *
 * \return pointer to dynamically allocated TVMValue array.
 */
uintptr_t* _TVMValueNativeAllocate(int n) {
  uintptr_t * ptr = reinterpret_cast<uintptr_t*>(calloc(n, sizeof(TVMValue)));
  return ptr;
}

/*!
 * \brief Native helper to copy TVMValue from golang slice to native array.
 * this helper is need as underlying momory for golang slice is not continueous.
 *
 * \param to_ptr is the native pointer of TVMValue array.
 * \param from_ptr pointer to TVMValue in golang slice.
 * \param array index in native array.
 */
void _TVMValueNativeSet(void* to_ptr, void* from_ptr, int ind) {
  TVMValue **from_p = reinterpret_cast<TVMValue**>(from_ptr);
  TVMValue *to_p = reinterpret_cast<TVMValue*>(to_ptr);
  memcpy(&(to_p[ind]), from_p, sizeof(TVMValue));
}

/*!
 * \brief Native helper to copy TVMValue from golang slice to native array.
 * this helper is need as underlying momory for golang slice is not continueous.
 *
 * \param to_ptr pointer to TVMValue in golang slice.
 * \param from_ptr is the native pointer of TVMValue array.
 * \param array index in native array.
 */
void _TVMValueNativeGet(void* to_ptr, void* from_ptr, int ind) {
  TVMValue *from_p = reinterpret_cast<TVMValue*>(from_ptr);
  TVMValue **to_p = reinterpret_cast<TVMValue**>(to_ptr);
  memcpy(to_p, &(from_p[ind]), sizeof(TVMValue));
}

/*!
 * \brief Native helper to free memory of TVMValue array.
 *
 * \param native_ptr is the pointer to native TVMValue array.
 */
void _TVMValueNativeFree(void * native_ptr) {
    free(native_ptr);
}

// Helpers for DLTensor

/*!
 * \brief Native helper to DLTensor aka TVMArray creation
 *
 * \return a new DLTensor aka TVMValue
 */
DLTensor *_NewDLTensor() {
  return reinterpret_cast<DLTensor *>(new DLTensor());
}

/*!
 * \brief Native helper to free DLTensor aka TVMArray
 *
 * \param dltensor is a valid DLTensor pointer.
 */
void _DeleteDLTensor(DLTensor *dltensor) {
  delete dltensor;
}

/*!
 * \brief Native helper to return data pointer of DLTensor
 *
 * \param dltensor is a valid DLTensor pointer.
 *
 * \return pointer to data inside DLTensor
 */
void *_DLTensorCopyTo(DLTensor *pdltensor) {
  return reinterpret_cast<void*>(pdltensor->data);
}

/*!
 * \brief Native helper to return ndim of DLTensor
 *
 * \param dltensor is a valid DLTensor pointer.
 *
 * \return ndim attribute inside DLTensor
 */
int _DLTensorGetNdim(DLTensor *pdltensor) {
  return pdltensor->ndim;
}

/*!
 * \brief Native helper to return shape attribute of DLTensor
 *
 * \param dltensor is a valid DLTensor pointer.
 *
 * \return shape attribute of DLTensor
 */
int64_t* _DLTensorGetShape(DLTensor *pdltensor) {
  return pdltensor->shape;
}

/*!
 * \brief Native helper to return dtype attribute of DLTensor
 *
 * \param dltensor is a valid DLTensor pointer.
 *
 * \return dtype of type TVMType.
 */
TVMType _DLTensorGetDType(DLTensor *pdltensor) {
  return pdltensor->dtype;
}

/*!
 * \brief Native helper to return ctx attribute of DLTensor
 *
 * \param dltensor is a valid DLTensor pointer.
 *
 * \return ctx of type TVMcontext inside DLTensor
 */
TVMContext _DLTensorGetCtx(DLTensor *pdltensor) {
  return pdltensor->ctx;
}

// Helpers for TVMByteArray

/*!
 * \brief Native helper to copy byte data from golang to native buffer.
 *
 * \param tbytearray is a valid TVMByteArray pointer.
 * \param val is pointer to data.
 * \param len is the data size.
 */
void _TVMByteArraySetData(TVMByteArray *tbytearray, char *val, int len) {
  if (tbytearray->data) {
    delete tbytearray->data;
  }
  tbytearray->data = val;
  tbytearray->size = (size_t)len;
}

/*!
 * \brief Native helper to return byte array from TVMByteArray.
 *
 * \param tbytearray is a valid TVMByteArray pointer.
 *
 * \return char pointer to array content.
 */
const char* _TVMByteArrayGetData(TVMByteArray *tbytearray) {
  return tbytearray->data;
}

/*!
 * \brief Native helper to return byte array from TVMByteArray.
 *
 * \param tbytearray is a valid TVMByteArray pointer.
 *
 * \return byte array length.
 */
int _TVMByteArrayGetDataLen(TVMByteArray *tbytearray) {
  return tbytearray->size;
}

/*!
 * \brief Native helper to create new TVMByteArray object.
 *
 * \return pointer to dynamically allocated TVMByteArray object.
 */
TVMByteArray *_NewTVMByteArray() {
  TVMByteArray *val =  reinterpret_cast<TVMByteArray *>(new TVMByteArray());
  val->data = NULL;
  val->size = 0;
  return val;
}

/*!
 * \brief Native helper to free TVMByteArray object.
 *
 * \param a valid pointer to TVMByteArray object.
 */
void _DeleteTVMByteArray(TVMByteArray *tbytearray) {
  if (tbytearray->data) {
    delete tbytearray->data;
  }
  delete tbytearray;
}

extern int goTVMCallback(void*, void*, int, void*, void*);

/*!
 * \brief _TVMCallback is the TVM runtime callback function for PackedFunction system.
 *
 * \param args is an array of TVMValue
 * \param type_codes is an array of int
 * \param num_args is int representing number of in arguments
 * \param ret is the return value handle to set the packed function return.
 * \param resource_handle is the golang private data pointer.
 *
 * \returns the error status as TVM_DLL
 */
int _TVMCallback(TVMValue* args,
                 int* type_codes,
                 int num_args,
                 TVMRetValueHandle ret,
                 void* resource_handle) {
    return goTVMCallback(args, type_codes, num_args, ret, resource_handle);
}

/*!
 * _TVMPackedCFuncFinalizer is finalizer for packed function system.
 *
 */
void _TVMPackedCFuncFinalizer(void* resource_handle) {
    return;
}

/*!
 * /brief _ConvertFunction creates a packed function for with given resource handle.
 *
 * /param fptr is the pointer to golang resource handle.
 * /param *fhandle is the return argument holding packed function.
 *
 * /return is an int indicating the return status.
 */
int _ConvertFunction(void* fptr, TVMFunctionHandle *fhandle) {
  int ret = TVMFuncCreateFromCFunc(_TVMCallback,
                                   fptr,
                                   _TVMPackedCFuncFinalizer,
                                   fhandle);
  return ret;
}

/*!
 * /brief _RegisterFunction is to register the packed function with GlobalFunction.
 *
 * /param fname is the name of the function to register.
 * /param fhandle is the packed function handle.
 *
 * /return is an int indicating return status ov TVM call.
 */
int _RegisterFunction(char *func_name, TVMFunctionHandle fhandle) {
  int ret = TVMFuncRegisterGlobal(func_name, fhandle, 0);  // Override = False
  _native_free(func_name);
  return ret;
}

#ifdef __cplusplus
}
#endif

