from ..base import TVMError
from libcpp.vector cimport vector
from cpython.version cimport PY_MAJOR_VERSION
from libc.stdint cimport int64_t, uint8_t, uint16_t
import ctypes

cdef enum TVMTypeCode:
    kInt = 0
    kUInt = 1
    kFloat = 2
    kHandle = 3
    kNull = 4
    kArrayHandle = 5
    kTVMType = 6
    kNodeHandle = 7
    kModuleHandle = 8
    kFuncHandle = 9
    kStr = 10
    kBytes = 11

cdef extern from "tvm/runtime/c_runtime_api.h":
    struct DLType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct TVMValue:
        int64_t v_int64
        double v_float64
        void* v_handle
        const char* v_str
        DLType v_type

ctypedef void* TVMRetValueHandle
ctypedef void* TVMFunctionHandle
ctypedef void* NodeHandle

ctypedef int (*TVMPackedCFunc)(
    TVMValue* args,
    int* type_codes,
    int num_args,
    TVMRetValueHandle ret,
    void* resource_handle)

ctypedef void (*TVMPackedCFuncFinalizer)(void* resource_handle)

cdef extern from "tvm/runtime/c_runtime_api.h":
    void TVMAPISetLastError(const char* msg);
    const char *TVMGetLastError();
    int TVMFuncCall(TVMFunctionHandle func,
                    TVMValue* arg_values,
                    int* type_codes,
                    int num_args,
                    TVMValue* ret_val,
                    int* ret_type_code)
    int TVMFuncFree(TVMFunctionHandle func)
    int TVMCFuncSetReturn(TVMRetValueHandle ret,
                          TVMValue value,
                          int type_code)
    int TVMFuncCreateFromCFunc(TVMPackedCFunc func,
                               void* resource_handle,
                               TVMPackedCFuncFinalizer fin,
                               TVMFunctionHandle *out)

cdef extern from "tvm/c_api.h":
    int TVMCbArgToReturn(TVMValue* value, int code)
    int TVMNodeFree(NodeHandle handle)
    TVMNodeTypeKey2Index(const char* type_key,
                         int* out_index)
    int TVMNodeGetTypeIndex(NodeHandle handle,
                            int* out_index)
    int TVMNodeGetAttr(NodeHandle handle,
                       const char* key,
                       TVMValue* out_value,
                       int* out_type_code,
                       int* out_success)

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
        raise TVMError(TVMGetLastError())


cdef inline object ctypes_handle(void* chandle):
    """Cast C handle to ctypes handle."""
    return ctypes.cast(<unsigned long long>chandle, ctypes.c_void_p)

cdef inline void* c_handle(object handle):
    """Cast C types handle to c handle."""
    cdef unsigned long long v_ptr
    v_ptr = handle.value
    return <void*>(v_ptr)
