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

from ..base import raise_last_ffi_error
from libcpp cimport bool as bool_t
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
    kDLDevice = 6
    kTVMDLTensorHandle = 7
    kTVMObjectHandle = 8
    kTVMModuleHandle = 9
    kTVMPackedFuncHandle = 10
    kTVMStr = 11
    kTVMBytes = 12
    kTVMNDArrayHandle = 13
    kTVMObjectRefArg = 14
    kTVMArgBool = 15
    kTVMExtBegin = 16

cdef extern from "tvm/runtime/c_runtime_api.h":
    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLDevice:
        int device_type
        int device_id

    ctypedef struct DLTensor:
        void* data
        DLDevice device
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
        bool_t v_bool
        double v_float64
        void* v_handle
        const char* v_str
        DLDataType v_type
        DLDevice v_device

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

# NOTE: All of TVM's C API function can be called without gil.
# for API functions that can be run long(e.g. FuncCall)
# we need to explicitly release the GIL as follows.
#
# cdef myfunc():
#     cdef int c_api_ret_code
#     with nogil:
#         c_api_ret_code = TVMAPIFunc(...)
#     CHECK_CALL(c_apt_ret_code)
#
# Explicitly releasing the GIL enables other python threads
# to continue running while we are in TVMAPIFunc.
# Not releasing GIL explicitly is OK(and perhaps desirable)
# for short-running functions, as frequent unlocking also takes time,
# the python interpreter will release GIL in a set period.
#
# We mark the possibly long running function as nogil below.
cdef extern from "tvm/runtime/c_runtime_api.h":
    void TVMAPISetLastError(const char* msg)
    void TVMAPISetLastPythonError(void* py_object) except +
    const char *TVMGetLastError()
    int TVMFuncGetGlobal(const char* name,
                         TVMPackedFuncHandle* out)
    int TVMFuncCall(TVMPackedFuncHandle func,
                    TVMValue* arg_values,
                    int* type_codes,
                    int num_args,
                    TVMValue* ret_val,
                    int* ret_type_code) nogil
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
                      DLDevice dev,
                      DLTensorHandle* out) nogil
    int TVMArrayFree(DLTensorHandle handle) nogil
    int TVMArrayCopyFromTo(DLTensorHandle src,
                           DLTensorHandle to,
                           TVMStreamHandle stream) nogil
    int TVMArrayFromDLPack(DLManagedTensor* arr_from,
                           DLTensorHandle* out) nogil
    int TVMArrayToDLPack(DLTensorHandle arr_from,
                         DLManagedTensor** out) nogil
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


cdef inline int CHECK_CALL(int ret) except -2:
    """Check the return code of the C API function call"""
    # -2 brings exception
    if ret == -2:
        return -2
    if ret != 0:
        raise_last_ffi_error()
    return 0


cdef inline object ctypes_handle(void* chandle):
    """Cast C handle to ctypes handle."""
    return ctypes.cast(<unsigned long long>chandle, ctypes.c_void_p)


cdef inline void* c_handle(object handle):
    """Cast C types handle to c handle."""
    cdef unsigned long long v_ptr
    v_ptr = handle.value
    return <void*>(v_ptr)


# python env API
cdef extern from "Python.h":
    int PyErr_CheckSignals()
    void* PyGILState_Ensure()
    void PyGILState_Release(void*)
    void Py_IncRef(void*)
    void Py_DecRef(void*)

cdef extern from "tvm/runtime/c_backend_api.h":
    int TVMBackendRegisterEnvCAPI(const char* name, void* ptr)

cdef _init_env_api():
    # Initialize env api for signal handling
    # so backend can call tvm::runtime::EnvCheckSignals to check
    # signal when executing a long running function.
    #
    # Also registers the gil state release and ensure as PyErr_CheckSignals
    # function is called with gil released and we need to regrab the gil
    CHECK_CALL(TVMBackendRegisterEnvCAPI(c_str("PyErr_CheckSignals"), <void*>PyErr_CheckSignals))
    CHECK_CALL(TVMBackendRegisterEnvCAPI(c_str("PyGILState_Ensure"), <void*>PyGILState_Ensure))
    CHECK_CALL(TVMBackendRegisterEnvCAPI(c_str("PyGILState_Release"), <void*>PyGILState_Release))
    CHECK_CALL(TVMBackendRegisterEnvCAPI(c_str("PyGILState_Release"), <void*>PyGILState_Release))
    CHECK_CALL(TVMBackendRegisterEnvCAPI(c_str("Py_IncRef"), <void*>Py_IncRef))
    CHECK_CALL(TVMBackendRegisterEnvCAPI(c_str("Py_DecRef"), <void*>Py_DecRef))

_init_env_api()
