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

// Memory allocation and free helpers
void * _native_malloc(int len) {
  return malloc(len);
}

void _native_free(void *ptr) {
  free (ptr);
}

// golang string to native char array conversion helpers
// This method allocates the char buffer which need to be freed by caller.
char * _gostring_to_native(_gostring_ gostr) {
  char *nstr;

  nstr = (char *)_native_malloc(gostr.n + 1);

  if(nstr) {
    memcpy(nstr, gostr.p, gostr.n);
    nstr[gostr.n] = '\0';
  }

  return nstr;
}

static _gostring_ _native_to_gostring(const char *p, size_t l) {
  _gostring_ ret;
  ret.p = (char*) _native_malloc(l+1);
  memcpy(ret.p, p, l+1);
  ret.n = l;
  return ret;
}

static void putuint64(std::string *s, size_t off, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        (*s)[off + i] = (v >> (i * 8)) & 0xff;
    }
}

// TVM runtime C interface wrappers

_gostring_ _TVM_VERSION(void) {
  char version[8];
  strcpy(version, TVM_VERSION);

  return _native_to_gostring(version, strlen(TVM_VERSION));
}

int _TVMFuncListGlobalNames(_gostring_* names) {
  int names_size ;
  char **names_array;
  int result; 

  result = (int)TVMFuncListGlobalNames(&names_size,(char const ***)&names_array);

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

_gostring_ _TVMGetLastError(void) {
  char *result;
    
  result = (char *)TVMGetLastError();
  return _native_to_gostring(result, result ? strlen(result) : 0);
}

int _TVMModLoadFromFile(_gostring_ modpath, _gostring_ modtype, TVMModuleHandle *modp) {
  char *nmodpath = _gostring_to_native(modpath);
  char *nmodtype = _gostring_to_native(modtype);

  if(!nmodpath) {
    if(nmodtype) _native_free(nmodtype);
    return -1;
  }

  if(!nmodtype) {
    if(nmodpath) _native_free(nmodpath);
    return -1;
  }

  int ret =  (int)TVMModLoadFromFile((char const *)nmodpath, (char const *)nmodtype, modp);

  _native_free(nmodpath);
  _native_free(nmodtype);
}

int _TVMFuncGetGlobal(_gostring_ funcname, TVMModuleHandle *funp) {
  char *nfuncname = _gostring_to_native(funcname);

  if(!nfuncname) {
    return -1;
  }

  int ret =  (int)TVMFuncGetGlobal((char const *)nfuncname, funp);

  _native_free(nfuncname);
}

int _TVMArrayAlloc(long long *shape, int ndim,
                   int dtype_code, int dtype_bits, int dtype_lanes,
                   int device_type, int device_id, TVMArrayHandle *dltensor) {
  return (int)TVMArrayAlloc((tvm_index_t *)shape, ndim,
                            dtype_code, dtype_bits, dtype_lanes,
                            device_type, device_id, dltensor);
}

int _TVMArrayFree(DLTensor *dltensor) {
  return (int)TVMArrayFree(dltensor);
}

int _TVMModGetFunction(TVMModuleHandle modp, _gostring_ funcname,
                       int query_imports, TVMFunctionHandle *funp) {
  char *nfuncname = _gostring_to_native(funcname);
  if(!nfuncname) {
    return -1;
  }
    
  int result = (int)TVMModGetFunction(modp, (char const *)nfuncname, query_imports, funp);

  _native_free(nfuncname);

  return result;
}

int _TVMFuncCall(TVMFunctionHandle funp, TVMValue *arg_values, int *type_codes,
                 int num_args, TVMValue *ret_values, int *ret_type_codes) {
  return TVMFuncCall(funp, arg_values, type_codes, num_args, ret_values, ret_type_codes);
}

// Helpers for TVMValue
TVMValue * _NewTVMValue() {
  return (TVMValue *)new TVMValue();
}
    
void _DeleteTVMValue(TVMValue *tvmval) {
  delete tvmval;
}   

void _TVMValueSetInt64(TVMValue *tvmval, long long val) {
  tvmval->v_int64 = val;
}

long long _TVMValueGetInt64(TVMValue *tvmval) {
  return tvmval->v_int64;
}

void _TVMValueSetFloat64(TVMValue *tvmval, double val) {
  tvmval->v_float64 = val;
}

double _TVMValueGetFloat64(TVMValue *tvmval) {
  return tvmval->v_float64;
}

void _TVMValueSetHandle(TVMValue *tvmval, void* val) {
  tvmval->v_handle = val;
}

void * _TVMValueGetHandle(TVMValue *tvmval) {
  return tvmval->v_handle;
}

void _TVMValueSetStr(TVMValue *tvmval, _gostring_ val) {
  tvmval->v_str = (char*) (new char[val.n+1]);
  strncpy((char *)tvmval->v_str, (const char *)val.p, val.n);
  ((char*)(tvmval->v_str))[val.n] = 0;
}

_gostring_ _TVMValueGetStr(TVMValue *tvmval) {
  return _native_to_gostring(tvmval->v_str, strlen(tvmval->v_str));
}

void _TVMValueUnSetStr(TVMValue *tvmval) {
  if (tvmval->v_str) {
    delete (tvmval->v_str);
    tvmval->v_str = 0;
  }
}

uintptr_t* _TVMValueNativeAllocate(int n) {
  uintptr_t * ptr= (uintptr_t*) calloc(n, sizeof(TVMValue));
  return ptr;
}

void _TVMValueNativeSet(void* to_ptr, void* from_ptr, int ind) {
  TVMValue **from_p = (TVMValue**) from_ptr;
  TVMValue *to_p = (TVMValue*) to_ptr;

  memcpy(&(to_p[ind]), from_p, sizeof(TVMValue));
}

void _TVMValueNativeGet(void* to_ptr, void* from_ptr, int ind) {
  TVMValue *from_p = (TVMValue*) from_ptr;
  TVMValue **to_p = (TVMValue**) to_ptr;

  memcpy(to_p, &(from_p[ind]), sizeof(TVMValue));
}

void _TVMValueNativeFree(void * native_ptr) {
    free(native_ptr);
}

// Helpers for DLTensor
DLTensor *_NewDLTensor() {
  return (DLTensor *)new DLTensor();
}


void _DeleteDLTensor(DLTensor *dltensor) {
  delete dltensor;
}

void *_DLTensorGetData(DLTensor *pdltensor) {
  return (void *) (pdltensor->data);
}

// Helpers for TVMByteArray
void _TVMByteArraySetData(TVMByteArray *tbytearray, _gostring_ val) {
  tbytearray->data = (char*) (new char[val.n]);
  memcpy((char *)tbytearray->data, (const char *)val.p, val.n);
}

_gostring_ _TVMByteArrayGetData(TVMByteArray *tbytearray) {
  // (size - 1) : _native_to_gostring assume null char at end.
  return _native_to_gostring(tbytearray->data, (tbytearray->size - 1));
}


void _TVMByteArraySetSize(TVMByteArray *tbytearray, long long val) {
  tbytearray->size = (size_t)val;
}


long long _TVMByteArrayGetSize(TVMByteArray *tbytearray) {
  return tbytearray->size;
}


TVMByteArray *_NewTVMByteArray() {
  return (TVMByteArray *)new TVMByteArray();
}


void _DeleteTVMByteArray(TVMByteArray *tbytearray) {
  delete tbytearray;
}


#ifdef __cplusplus
}
#endif

