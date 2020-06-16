/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief gotvm native interface definition
 * \file gotvm.cxx
 */

// Standard includes
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

// golang string compatible definition
typedef struct {
  char* p;
  int n;
} _gostring_;
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

// TVM runtime C interface
#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>

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
static _gostring_ _native_to_gostring(const char* p, size_t l) {
  _gostring_ ret;
  ret.p = reinterpret_cast<char*>(malloc(l));
  if (NULL == ret.p) {
    ret.n = 0;
    return ret;
  }
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
static void putuint64(std::string* s, size_t off, uint64_t v) {
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
  const char* version = TVM_VERSION;
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
  char** names_array;
  int result;

  result = TVMFuncListGlobalNames(&names_size, (char const***)&names_array);
  if (result) {
    return result;
  }

  size_t tot = 8;
  for (int ii = 0; ii < names_size; ++ii) {
    tot += 8 + strlen(names_array[ii]);
  }

  std::string str;
  str.resize(tot);
  putuint64(&str, 0, names_size);
  size_t off = 8;
  for (int64_t ii = 0; ii < names_size; ++ii) {
    putuint64(&str, off, strlen(names_array[ii]));
    off += 8;
    str.replace(off, strlen(names_array[ii]), names_array[ii]);
    off += strlen(names_array[ii]);
  }
  *names = _native_to_gostring(str.data(), str.size());
  if (str.size() != names->n) {
    TVMAPISetLastError("malloc failed during _native_to_gostring");
    result = 1;
  }
  return result;
}

// Helpers for TVMValue

/*!
 * \brief Native helper to copy TVMValue from golang slice to native array.
 * this helper is need as underlying momory for golang slice is not continueous.
 *
 * \param to_ptr is the native pointer of TVMValue array.
 * \param from_ptr pointer to TVMValue in golang slice.
 * \param array index in native array.
 */
void _TVMValueNativeSet(void* to_ptr, void* from_ptr, int ind) {
  TVMValue* from_p = reinterpret_cast<TVMValue*>(from_ptr);
  TVMValue* to_p = reinterpret_cast<TVMValue*>(to_ptr);
  memcpy(to_p + ind, from_p, sizeof(TVMValue));
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
  TVMValue* from_p = reinterpret_cast<TVMValue*>(from_ptr);
  TVMValue* to_p = reinterpret_cast<TVMValue*>(to_ptr);
  memcpy(to_p, from_p + ind, sizeof(TVMValue));
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
int _TVMCallback(TVMValue* args, int* type_codes, int num_args, TVMRetValueHandle ret,
                 void* resource_handle) {
  return goTVMCallback(args, type_codes, num_args, ret, resource_handle);
}

/*!
 * _TVMPackedCFuncFinalizer is finalizer for packed function system.
 *
 */
void _TVMPackedCFuncFinalizer(void* resource_handle) { return; }

/*!
 * /brief _ConvertFunction creates a packed function for with given resource handle.
 *
 * /param fptr is the pointer to golang resource handle.
 * /param *fhandle is the return argument holding packed function.
 *
 * /return is an int indicating the return status.
 */
int _ConvertFunction(void* fptr, TVMFunctionHandle* fhandle) {
  int ret = TVMFuncCreateFromCFunc(_TVMCallback, fptr, _TVMPackedCFuncFinalizer, fhandle);
  return ret;
}

#ifdef __cplusplus
}
#endif
