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
from libc.stdint cimport int32_t, int64_t, uint64_t, uint32_t, uint8_t, int16_t
from libc.string cimport memcpy
from libcpp.vector cimport vector
from cpython.bytes cimport PyBytes_AsStringAndSize, PyBytes_FromStringAndSize, PyBytes_AsString
from cpython cimport Py_INCREF, Py_DECREF
from cpython cimport PyErr_CheckSignals, PyGILState_Ensure, PyGILState_Release, PyObject
from cpython cimport pycapsule, PyCapsule_Destructor
from cpython cimport PyErr_SetNone

# Cython internal for custom traceback creation
cdef extern from *:
    """
    static void __Pyx_AddTraceback(const char *funcname, int c_line, int py_line, const char *filename);
    static void __Pyx_AddTraceback_(const char *funcname, int c_line, int py_line, const char *filename) {
      __Pyx_AddTraceback(funcname, c_line, py_line, filename);
    }
    """
    # __Pyx_AddTraceback is a Cython internal function to add custom traceback to exception
    # We declare __Pyx_AddTraceback_ and redirect to it to avoid mixed C/C++ calling
    # This is a nice hack to enable us to create customized traceback refers to c++ code
    void __Pyx_AddTraceback_(const char *funcname, int c_line, int py_line, const char *filename)

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
        kTVMFFIFunction = 68
        kTVMFFIArray = 69
        kTVMFFIMap = 70
        kTVMFFIShape = 71
        kTVMFFINDArray = 72
        kTVMFFIModule = 73

    ctypedef void* TVMFFIObjectHandle

    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        int16_t lanes

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

    ctypedef struct DLPackVersion:
        uint32_t major
        uint32_t minor

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void* manager_ctx
        void (*deleter)(DLManagedTensor* self)

    ctypedef struct DLManagedTensorVersioned:
        DLPackVersion version
        DLManagedTensor dl_tensor
        void* manager_ctx
        void (*deleter)(DLManagedTensorVersioned* self)
        uint64_t flags

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
        size_t size

    ctypedef struct TVMFFIShapeArray:
        const int64_t* data
        size_t size

    ctypedef struct TVMFFIErrorInfo:
        const char* kind
        const char* message
        const char* traceback

    ctypedef int (*TVMFFISafeCallType)(
        void* ctx, const TVMFFIAny* args, int32_t num_args,
        TVMFFIAny* result) noexcept

    int TVMFFIObjectFree(TVMFFIObjectHandle obj) nogil
    int TVMFFIObjectGetTypeIndex(TVMFFIObjectHandle obj) nogil
    int TVMFFIFunctionCall(TVMFFIObjectHandle func, TVMFFIAny* args, int32_t num_args,
                           TVMFFIAny* result) nogil
    int TVMFFIFunctionCreate(void* self, TVMFFISafeCallType safe_call,
                         void (*deleter)(void*), TVMFFIObjectHandle* out) nogil
    int TVMFFIAnyViewToOwnedAny(const TVMFFIAny* any_view, TVMFFIAny* out) nogil
    int TVMFFIFunctionSetGlobal(const char* name, TVMFFIObjectHandle f, int override) nogil
    int TVMFFIFunctionGetGlobal(const char* name, TVMFFIObjectHandle* out) nogil
    void TVMFFIErrorMoveFromRaised(TVMFFIObjectHandle* result) nogil
    void TVMFFIErrorSetRaised(TVMFFIObjectHandle error) nogil
    void TVMFFIErrorSetRaisedCStr(const char* kind, const char* message) nogil
    void TVMFFIErrorUpdateTraceback(TVMFFIObjectHandle error, const char* traceback) nogil
    int TVMFFIErrorCreate(const char* kind, const char* message, const char* traceback,
                          TVMFFIObjectHandle* out) nogil
    int TVMFFIEnvRegisterCAPI(const char* name, void* ptr) nogil
    int TVMFFITypeKeyToIndex(const char* type_key, int32_t* out_tindex) nogil
    int TVMFFIDataTypeFromString(const char* str, DLDataType* out) nogil
    int TVMFFIDataTypeToString(DLDataType dtype, TVMFFIObjectHandle* out) nogil
    const char* TVMFFITraceback(const char* filename, int lineno, const char* func) nogil;
    int TVMFFINDArrayFromDLPack(DLManagedTensor* src, int32_t require_alignment,
                                int32_t require_contiguous, TVMFFIObjectHandle* out) nogil
    int TVMFFINDArrayFromDLPackVersioned(DLManagedTensorVersioned* src,
                                        int32_t require_alignment,
                                        int32_t require_contiguous,
                                        TVMFFIObjectHandle* out) nogil
    int TVMFFINDArrayToDLPack(TVMFFIObjectHandle src, DLManagedTensor** out) nogil
    int TVMFFINDArrayToDLPackVersioned(TVMFFIObjectHandle src,
                                        DLManagedTensorVersioned** out) nogil
    TVMFFIByteArray* TVMFFIBytesGetByteArrayPtr(TVMFFIObjectHandle obj) nogil
    TVMFFIErrorInfo* TVMFFIErrorGetErrorInfoPtr(TVMFFIObjectHandle obj) nogil
    TVMFFIShapeArray* TVMFFIShapeGetShapeArrayPtr(TVMFFIObjectHandle obj) nogil
    DLTensor* TVMFFINDArrayGetDLTensorPtr(TVMFFIObjectHandle obj) nogil
    DLDevice TVMFFIDLDeviceFromIntPair(int32_t device_type, int32_t device_id) nogil


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


cdef _init_env_api():
    # Initialize env api for signal handling
    # Also registers the gil state release and ensure as PyErr_CheckSignals
    # function is called with gil released and we need to regrab the gil
    CHECK_CALL(TVMFFIEnvRegisterCAPI(c_str("PyErr_CheckSignals"), <void*>PyErr_CheckSignals))
    CHECK_CALL(TVMFFIEnvRegisterCAPI(c_str("PyGILState_Ensure"), <void*>PyGILState_Ensure))
    CHECK_CALL(TVMFFIEnvRegisterCAPI(c_str("PyGILState_Release"), <void*>PyGILState_Release))

_init_env_api()
