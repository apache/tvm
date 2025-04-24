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
import ctypes
from libc.stdint cimport int32_t, int64_t, uint64_t, uint32_t, uint8_t, uint16_t
from libc.string cimport memcpy
from libcpp.vector cimport vector

# Cython binding for TVM FFI C API
cdef extern from "tvm/ffi/c_api.h":
    cdef enum TVMFFITypeIndex:
        kTVMFFIAny = -1
        kTVMFFINone = 0
        kTVMFFIInt = 1
        kTVMFFIBool = 2
        kTVMFFIFloat = 3
        kTVMFFIOpaquePtr = 4
        kTVMFFIDataType = 5
        kTVMFFIDevice = 6
        kTVMFFIDLTensorPtr = 7
        kTVMFFIRawStr = 8
        kTVMFFIByteArrayPtr = 9
        kTVMFFIObjectRValueRef = 10
        kTVMFFIStaticObjectBegin = 64
        kTVMFFIObject = 64
        kTVMFFIStr = 65
        kTVMFFIBytes = 66
        kTVMFFIError = 67
        kTVMFFIFunc = 68
        kTVMFFIArray = 69
        kTVMFFIMap = 70
        kTVMFFIShape = 71
        kTVMFFINDArray = 72
        kTVMFFIModule = 73

    ctypedef void* TVMFFIObjectHandle

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

    ctypedef struct TVMFFIObject:
        int32_t type_index
        int32_t ref_counter
        void (*deleter)(TVMFFIObject* self)

    ctypedef struct TVMFFIAny:
        int32_t type_index
        int32_t padding
        int64_t v_int64
        double v_float64
        void* v_ptr
        TVMFFIObject* v_obj
        const char* v_c_str
        DLDataType v_dtype
        DLDevice v_device

    ctypedef struct TVMFFIByteArray:
        const char* data
        int64_t size

    ctypedef int (*TVMFFISafeCallType)(
        void* self, const TVMFFIAny* args, int32_t num_args,
        TVMFFIAny* result)

    int TVMFFIObjectFree(TVMFFIObjectHandle obj) nogil
    int TVMFFIObjectGetTypeIndex(TVMFFIObjectHandle obj) nogil
    int TVMFFIFuncCall(TVMFFIObjectHandle func, TVMFFIAny* args, int32_t num_args,
                       TVMFFIAny* result) nogil
    int TVMFFIFuncCreate(void* self, TVMFFISafeCallType safe_call,
                         void (*deleter)(void*), TVMFFIObjectHandle* out) nogil
    int TVMFFIFuncSetGlobal(const char* name, TVMFFIObjectHandle f, int override) nogil
    int TVMFFIFuncGetGlobal(const char* name, TVMFFIObjectHandle* out) nogil
    void TVMFFIMoveFromLastError(TVMFFIAny* result) nogil
    void TVMFFISetLastError(const TVMFFIAny* error_view) nogil
    int TVMFFITypeKeyToIndex(const char* type_key, int32_t* out_tindex) nogil
    int TVMFFIDataTypeFromString(const char* str, DLDataType* out) nogil
    int TVMFFIDataTypeToString(DLDataType dtype, TVMFFIObjectHandle* out) nogil
    TVMFFIByteArray* TVMFFIBytesGetByteArrayPtr(TVMFFIObjectHandle obj) nogil


cdef inline int raise_last_ffi_error() except -2:
    """Raise the last FFI error as a Python exception"""
    # TODO: Implement
    return 0


cdef inline int CHECK_CALL(int ret) except -2:
    """Check the return code of the C API function call"""
    # -2 brings exception
    if ret == -2:
        return -2
    if ret != 0:
        raise_last_ffi_error()
    return 0


cdef inline py_str(const char* x):
    """Convert a c_char_p to a python string

    Parameters
    ----------
    x : c_char_p
        A char pointer that can be passed to C API
    """
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


cdef inline object ctypes_handle(void* chandle):
    """Cast C handle to ctypes handle."""
    return ctypes.cast(<unsigned long long>chandle, ctypes.c_void_p)


cdef inline void* c_handle(object handle):
    """Cast C types handle to c handle."""
    cdef unsigned long long v_ptr
    v_ptr = handle.value
    return <void*>(v_ptr)
