# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from ..base import get_last_ffi_error
from libcpp.vector cimport vector
from cpython.version cimport PY_MAJOR_VERSION
from cpython cimport pycapsule
from libc.stdint cimport int32_t, int64_t, uint64_t, uint32_t, uint8_t, uint16_t
import ctypes

cdef enum TVMArgTypeCode:
    kInt = 0
    kUInt = 1
    kFloat = 2
    kTVMOpaqueHandle = 3
    kTVMNullptr = 4
    kTVMDataType = 5
    kTVMContext = 6
    kTVMDLTensorHandle = 7
    kTVMObjectHandle = 8
    kTVMModuleHandle = 9
    kTVMPackedFuncHandle = 10
    kTVMStr = 11
    kTVMBytes = 12
    kTVMNDArrayHandle = 13
    kTVMObjectRefArg = 14
    kTVMExtBegin = 15

cdef extern from "tvm/runtime/c_runtime_api.h":
    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLContext:
        int device_type
        int device_id

    ctypedef struct DLTensor:
        void* data
        DLContext ctx
        int ndim
        DLDataType dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void* manager_ctx
        void (*deleter)(DLManagedTensor* self)

    ctypedef struct TVMValue:
        int64_t v_int64
        double v_float64
        void* v_handle
        const char* v_str
        DLDataType v_type
        DLContext v_ctx

ctypedef int64_t tvm_index_t
ctypedef DLTensor* DLTensorHandle
ctypedef void* TVMStreamHandle
ctypedef void* TVMRetValueHandle
ctypedef void* TVMPackedFuncHandle
ctypedef void* ObjectHandle

ctypedef struct TVMObject:
    uint32_t type_index_
    int32_t ref_counter_
    void (*deleter_)(TVMObject* self)


ctypedef int (*TVMPackedCFunc)(
    TVMValue* args,
    int* type_codes,
    int num_args,
    TVMRetValueHandle ret,
    void* resource_handle)

ctypedef void (*TVMPackedCFuncFinalizer)(void* resource_handle)

cdef extern from "tvm/runtime/c_runtime_api.h":
    void TVMAPISetLastError(const char* msg)
    const char *TVMGetLastError()
    int TVMFuncGetGlobal(const char* name,
                         TVMPackedFuncHandle* out);
    int TVMFuncCall(TVMPackedFuncHandle func,
                    TVMValue* arg_values,
                    int* type_codes,
                    int num_args,
                    TVMValue* ret_val,
                    int* ret_type_code)
    int TVMFuncFree(TVMPackedFuncHandle func)
    int TVMCFuncSetReturn(TVMRetValueHandle ret,
                          TVMValue* value,
                          int* type_code,
                          int num_ret)
    int TVMFuncCreateFromCFunc(TVMPackedCFunc func,
                               void* resource_handle,
                               TVMPackedCFuncFinalizer fin,
                               TVMPackedFuncHandle *out)
    int TVMCbArgToReturn(TVMValue* value, int* code)
    int TVMArrayAlloc(tvm_index_t* shape,
                      tvm_index_t ndim,
                      DLDataType dtype,
                      DLContext ctx,
                      DLTensorHandle* out)
    int TVMArrayFree(DLTensorHandle handle)
    int TVMArrayCopyFromTo(DLTensorHandle src,
                           DLTensorHandle to,
                           TVMStreamHandle stream)
    int TVMArrayFromDLPack(DLManagedTensor* arr_from,
                           DLTensorHandle* out)
    int TVMArrayToDLPack(DLTensorHandle arr_from,
                         DLManagedTensor** out)
    void TVMDLManagedTensorCallDeleter(DLManagedTensor* dltensor)
    int TVMObjectFree(ObjectHandle obj)
    int TVMObjectGetTypeIndex(ObjectHandle obj, unsigned* out_index)


cdef inline py_str(const char* x):
    if PY_MAJOR_VERSION < 3:
        return x
    else:
        return x.decode("utf-8")


cdef inline c_str(pystr):
    """Create ctypes char * from a python string
    Parameters
    ----------
    string : string type
        python string

    Returns
    -------
    str : c_char_p
        A char pointer that can be passed to C API
    """
    return pystr.encode("utf-8")


cdef inline CALL(int ret):
    if ret != 0:
        raise get_last_ffi_error()


cdef inline object ctypes_handle(void* chandle):
    """Cast C handle to ctypes handle."""
    return ctypes.cast(<unsigned long long>chandle, ctypes.c_void_p)


cdef inline void* c_handle(object handle):
    """Cast C types handle to c handle."""
    cdef unsigned long long v_ptr
    v_ptr = handle.value
    return <void*>(v_ptr)
