/*!
 *  Copyright (c) 2017 by Contributors
 * \brief SWIG Interface definition
 * \file tvmgo.i
 */

%module tvmgo
#define TVMGO_VERSION "0.1"

%{
#include <string>
#include <tvm/runtime/c_runtime_api.h>
#include <dlpack/dlpack.h>
%}

%include "stdint.i"
%include "cpointer.i"
%include "typemaps.i"

%typemap(gotype) (int) "int32"

/* typemap for TVMFuncListGlobalNames
 *
 * typemap it to go slice of strings
 */

%go_import("unsafe", "encoding/binary")

%inline
%{

static void putuint64(std::string *s, size_t off, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        (*s)[off + i] = (v >> (i * 8)) & 0xff;
    }
}

%}

%typemap(gotype) (int* out_size, const char*** out_array) "*[]string"

%typemap(imtype) (int* out_size, const char*** out_array) "*string"

%typemap(goin) (int* out_size, const char*** out_array)
%{
    var str string
    $result = &str
%}

%typemap(in) (int* out_size, const char*** out_array) (int out_size, char **out_array)
%{
  $1 = &out_size;
  $2 = &out_array;
%}

%typemap(argout,fragment="AllocateString") (int* out_size, const char*** out_array)
%{
  {
    size_t tot = 8;
    for (int ii = 0; ii < *$1 ; ++ii) {
      tot += 8 + strlen((*$2)[ii]);
    }
    std::string str;
    str.resize(tot);
    putuint64(&str, 0, *$1);
    size_t off = 8;
    for (int64_t ii = 0; ii < *$1 ; ++ii) {
      putuint64(&str, off, strlen((*$2)[ii]));
      off += 8;
      str.replace(off, strlen((*$2)[ii]), (*$2)[ii]);
      off += strlen((*$2)[ii]);
    }
    *$input = Swig_AllocateString(str.data(), str.size());
  }
%}

%typemap(goargout,fragment="CopyString") (int* out_size, const char*** out_array)
%{
    {
        str := swigCopyString(*$input)
        bin := binary.LittleEndian
        size := bin.Uint64([]byte(str[:8]))
        str = str[8:]
        r := make([]string, size)
        for i := range r {
            len := bin.Uint64([]byte(str[:8]))
            str = str[8:]
            r[i] = str[:len]
            str = str[len:]
        }
        *$1 = r
    }
%}

/* typemap for TVMFuncListGlobalNames - END */

/* typemap for TVMValue* (pointer to a TVMValue array)
 *
 * Go use slice of TVMValue which inturn converted to
 * dynamically allocated continueous TVMValue array.
 *
 */
%inline
%{

uintptr_t* TVMValueNativeAllocate(int n) {
  uintptr_t * ptr= (uintptr_t*) calloc(n, sizeof(TVMValue));
  return ptr;
}

void TVMValueNativeSet(void* to_ptr, void* from_ptr, int ind) {
  TVMValue **from_p = (TVMValue**) from_ptr;
  TVMValue *to_p = (TVMValue*) to_ptr;

  memcpy(&(to_p[ind]), from_p, sizeof(TVMValue));
}

void TVMValueNativeGet(void* to_ptr, void* from_ptr, int ind) {
  TVMValue *from_p = (TVMValue*) from_ptr;
  TVMValue **to_p = (TVMValue**) to_ptr;

  memcpy(to_p, &(from_p[ind]), sizeof(TVMValue));
}

void TVMValueNativeFree(void * native_ptr) {
    free(native_ptr);
}

%}
%typemap(gotype) (TVMValue* arg_values) "[]TVMValue"

%typemap(imtype) (TVMValue* arg_values) "uintptr"

%typemap(goin) (TVMValue* arg_values)
%{
    native_ptr$argnum := TVMValueNativeAllocate(int32(len($input)))
    argval$argnum := $input

    for ii := range $input {
        TVMValueNativeSet(uintptr(unsafe.Pointer(native_ptr$argnum)),
                          uintptr(unsafe.Pointer($input[ii].Swigcptr())),
                          int32(ii))
    }

    $result = (uintptr)(unsafe.Pointer(native_ptr$argnum))
%}

%typemap(goargout) (TVMValue* arg_values)
%{
    for ii := range argval$argnum {
        TVMValueNativeGet(uintptr(unsafe.Pointer(argval$argnum[ii].Swigcptr())),
                          uintptr(unsafe.Pointer(native_ptr$argnum)),
                          int32(ii))
    }

    TVMValueNativeFree(uintptr(unsafe.Pointer(native_ptr$argnum)))
%}

%apply (TVMValue* arg_values) { (TVMValue* ret_val) };
/* typemap for TVMValue* (pointer to a TVMValue array) - END */

/*
 * typemap for DLTensor
 * Typemap does freeing the allocated local DLTensor and
 * replace with the one returned from TVMArrayAlloc
 */
%typemap(gotype) (TVMArrayHandle* out) "*DLTensor"

%typemap(imtype) (TVMArrayHandle* out) "*uintptr"

%typemap(goin) (TVMArrayHandle* out)
%{
    DeleteDLTensor(*$input)
    var newptr$argnum uintptr

    $result = &newptr$argnum
%}

%typemap(goargout) (TVMArrayHandle* out)
%{
    *$1 = (DLTensor)(SwigcptrDLTensor(newptr$argnum))
%}

/* typemap for DLTensor - END */

/* typemap TVMByteArray */
%typemap(memberin) const char* data {
    /* Hack: accessing _swig_go_1 as we know it*/
    if ($input) {
      $1 = (const char*) (new char[_swig_go_1.n]);
      memcpy((char *)$1, (const char*)$input, _swig_go_1.n);
    } else {
      $1 = 0;
    }
}
/* typemap TVMByteArray - END */

%include <tvm/runtime/c_runtime_api.h>
%include <dlpack/dlpack.h>
