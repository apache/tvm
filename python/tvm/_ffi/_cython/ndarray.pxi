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

from ..runtime_ctypes import TVMArrayHandle
from cpython cimport PyCapsule_Destructor

cdef const char* _c_str_dltensor = "dltensor"
cdef const char* _c_str_used_dltensor = "used_dltensor"


cdef void _c_dlpack_deleter(object pycaps):
    cdef DLManagedTensor* dltensor
    if pycapsule.PyCapsule_IsValid(pycaps, _c_str_dltensor):
        dltensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(pycaps, _c_str_dltensor)
        TVMDLManagedTensorCallDeleter(dltensor)


def _from_dlpack(object dltensor):
    cdef DLManagedTensor* ptr
    cdef DLTensorHandle chandle
    cdef int c_api_ret_code
    if pycapsule.PyCapsule_IsValid(dltensor, _c_str_dltensor):
        ptr = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(dltensor, _c_str_dltensor)
        with nogil:
            c_api_ret_code = TVMArrayFromDLPack(ptr, &chandle)
        CHECK_CALL(c_api_ret_code)
        # set name and destructor to be empty
        pycapsule.PyCapsule_SetDestructor(dltensor, NULL)
        pycapsule.PyCapsule_SetName(dltensor, _c_str_used_dltensor)
        return c_make_array(chandle, False, False)
    raise ValueError("Expect a dltensor field, pycapsule.PyCapsule can only be consumed once")


cdef class NDArrayBase:
    cdef DLTensor* chandle
    cdef int c_is_view

    cdef inline _set_handle(self, handle):
        cdef unsigned long long ptr
        if handle is None:
            self.chandle = NULL
        else:
            ptr = ctypes.cast(handle, ctypes.c_void_p).value
            self.chandle = <DLTensor*>(ptr)

    property _tvm_handle:
        def __get__(self):
            return <unsigned long long>self.chandle

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return ctypes.cast(
                    <unsigned long long>self.chandle, TVMArrayHandle)

        def __set__(self, value):
            self._set_handle(value)

    property is_view:
        def __get__(self):
            return self.c_is_view != 0

    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.chandle.shape[i] for i in range(self.chandle.ndim))

    def __init__(self, handle, is_view):
        self._set_handle(handle)
        self.c_is_view = is_view

    def __dealloc__(self):
        cdef int c_api_ret_code
        if self.c_is_view == 0:
            with nogil:
                c_api_ret_code = TVMArrayFree(self.chandle)
            CHECK_CALL(c_api_ret_code)

    def _copyto(self, target_nd):
        """Internal function that implements copy to target ndarray."""
        cdef int c_api_ret_code
        with nogil:
            c_api_ret_code = TVMArrayCopyFromTo(self.chandle, (<NDArrayBase>target_nd).chandle, NULL)
        CHECK_CALL(c_api_ret_code)
        return target_nd

    def to_dlpack(self):
        """Produce an array from a DLPack Tensor without copying memory

        Returns
        -------
        dlpack : DLPack tensor view of the array data
        """
        cdef DLManagedTensor* dltensor
        cdef int c_api_ret_code
        if self.c_is_view != 0:
            raise ValueError("to_dlpack do not work with memory views")
        with nogil:
            c_api_ret_code = TVMArrayToDLPack(self.chandle, &dltensor)
        CHECK_CALL(c_api_ret_code)
        return pycapsule.PyCapsule_New(dltensor, _c_str_dltensor, <PyCapsule_Destructor>_c_dlpack_deleter)


# Import limited object-related function from C++ side to improve the speed
# NOTE: can only use POD-C compatible object in FFI.
cdef extern from "tvm/runtime/ndarray.h" namespace "tvm::runtime":
    cdef void* TVMArrayHandleToObjectHandle(DLTensorHandle handle)


cdef c_make_array(void* chandle, is_view, is_container):
    global _TVM_ND_CLS

    if is_container:
        tindex = (
            <TVMObject*>TVMArrayHandleToObjectHandle(<DLTensorHandle>chandle)).type_index_
        if tindex < len(_TVM_ND_CLS):
            cls = _TVM_ND_CLS[tindex]
            if cls is not None:
                ret = cls.__new__(cls)
            else:
                ret = _CLASS_NDARRAY.__new__(_CLASS_NDARRAY)
        else:
            ret = _CLASS_NDARRAY.__new__(_CLASS_NDARRAY)
        (<NDArrayBase>ret).chandle = <DLTensor*>chandle
        (<NDArrayBase>ret).c_is_view = <int>is_view
        return ret
    else:
        ret = _CLASS_NDARRAY.__new__(_CLASS_NDARRAY)
        (<NDArrayBase>ret).chandle = <DLTensor*>chandle
        (<NDArrayBase>ret).c_is_view = <int>is_view
        return ret


cdef _TVM_COMPATS = ()

cdef _TVM_EXT_RET = {}

def _reg_extension(cls, fcreate):
    global _TVM_COMPATS
    _TVM_COMPATS += (cls,)
    if fcreate:
        _TVM_EXT_RET[cls._tvm_tcode] = fcreate

cdef list _TVM_ND_CLS = []

cdef _register_ndarray(int index, object cls):
    """register object class"""
    global _TVM_ND_CLS
    while len(_TVM_ND_CLS) <= index:
        _TVM_ND_CLS.append(None)

    _TVM_ND_CLS[index] = cls


def _make_array(handle, is_view, is_container):
    cdef unsigned long long ptr
    ptr = ctypes.cast(handle, ctypes.c_void_p).value
    return c_make_array(<void*>ptr, is_view, is_container)

cdef object _CLASS_NDARRAY = None

def _set_class_ndarray(cls):
    global _CLASS_NDARRAY
    _CLASS_NDARRAY = cls
