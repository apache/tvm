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
import traceback
from cpython cimport Py_INCREF, Py_DECREF
from numbers import Number, Integral
from ..base import string_types, py2cerror
from ..runtime_ctypes import DataType, TVMContext, TVMByteArray, ObjectRValueRef


cdef void tvm_callback_finalize(void* fhandle):
    local_pyfunc = <object>(fhandle)
    Py_DECREF(local_pyfunc)

cdef int tvm_callback(TVMValue* args,
                      int* type_codes,
                      int num_args,
                      TVMRetValueHandle ret,
                      void* fhandle) with gil:
    cdef list pyargs
    cdef TVMValue value
    cdef int tcode
    local_pyfunc = <object>(fhandle)
    pyargs = []
    for i in range(num_args):
        value = args[i]
        tcode = type_codes[i]
        if (tcode == kTVMObjectHandle or
            tcode == kTVMPackedFuncHandle or
            tcode == kTVMModuleHandle or
            tcode == kTVMObjectRefArg or
            tcode > kTVMExtBegin):
            CALL(TVMCbArgToReturn(&value, &tcode))

        if tcode != kTVMDLTensorHandle:
            pyargs.append(make_ret(value, tcode))
        else:
            pyargs.append(c_make_array(value.v_handle, True, False))
    try:
        rv = local_pyfunc(*pyargs)
    except Exception:
        msg = traceback.format_exc()
        msg = py2cerror(msg)
        TVMAPISetLastError(c_str(msg))
        return -1
    if rv is not None:
        if isinstance(rv, tuple):
            raise ValueError("PackedFunction can only support one return value")
        temp_args = []
        make_arg(rv, &value, &tcode, temp_args)
        CALL(TVMCFuncSetReturn(ret, &value, &tcode, 1))
    return 0


cdef object make_packed_func(TVMPackedFuncHandle chandle, int is_global):
    obj = _CLASS_PACKED_FUNC.__new__(_CLASS_PACKED_FUNC)
    (<PackedFuncBase>obj).chandle = chandle
    (<PackedFuncBase>obj).is_global = is_global
    return obj


def convert_to_tvm_func(object pyfunc):
    """Convert a python function to TVM function

    Parameters
    ----------
    pyfunc : python function
        The python function to be converted.

    Returns
    -------
    tvmfunc: tvm.Function
        The converted tvm function.
    """
    cdef TVMPackedFuncHandle chandle
    Py_INCREF(pyfunc)
    CALL(TVMFuncCreateFromCFunc(tvm_callback,
                                <void*>(pyfunc),
                                tvm_callback_finalize,
                                &chandle))
    return make_packed_func(chandle, False)


cdef inline int make_arg(object arg,
                         TVMValue* value,
                         int* tcode,
                         list temp_args) except -1:
    """Pack arguments into c args tvm call accept"""
    cdef unsigned long long ptr
    if isinstance(arg, ObjectBase):
        value[0].v_handle = (<ObjectBase>arg).chandle
        tcode[0] = kTVMObjectHandle
    elif isinstance(arg, NDArrayBase):
        value[0].v_handle = (<NDArrayBase>arg).chandle
        tcode[0] = (kTVMNDArrayHandle if
                    not (<NDArrayBase>arg).c_is_view else kTVMDLTensorHandle)
    elif isinstance(arg, PyNativeObject):
        value[0].v_handle = (<ObjectBase>(arg.__tvm_object__)).chandle
        tcode[0] = kTVMObjectHandle
    elif isinstance(arg, _TVM_COMPATS):
        ptr = arg._tvm_handle
        value[0].v_handle = (<void*>ptr)
        tcode[0] = arg.__class__._tvm_tcode
    elif isinstance(arg, (int, long)):
        value[0].v_int64 = arg
        tcode[0] = kInt
    elif isinstance(arg, float):
        value[0].v_float64 = arg
        tcode[0] = kFloat
    elif isinstance(arg, str):
        tstr = c_str(arg)
        value[0].v_str = tstr
        tcode[0] = kTVMStr
        temp_args.append(tstr)
    elif arg is None:
        value[0].v_handle = NULL
        tcode[0] = kTVMNullptr
    elif isinstance(arg, Number):
        value[0].v_float64 = arg
        tcode[0] = kFloat
    elif isinstance(arg, DataType):
        tstr = c_str(str(arg))
        value[0].v_str = tstr
        tcode[0] = kTVMStr
        temp_args.append(tstr)
    elif isinstance(arg, TVMContext):
        value[0].v_ctx = (<DLContext*>(
            <unsigned long long>ctypes.addressof(arg)))[0]
        tcode[0] = kTVMContext
    elif isinstance(arg, (bytes, bytearray)):
        # from_buffer only taeks in bytearray.
        if isinstance(arg, bytes):
            byte_arr = bytearray(arg)
            temp_args.append(byte_arr)
            arg = byte_arr

        arr = TVMByteArray()
        arr.data = ctypes.cast(
            (ctypes.c_byte * len(arg)).from_buffer(arg),
            ctypes.POINTER(ctypes.c_byte))
        arr.size = len(arg)
        value[0].v_handle = <void*>(
            <unsigned long long>ctypes.addressof(arr))
        tcode[0] = kTVMBytes
        temp_args.append(arr)
    elif isinstance(arg, string_types):
        tstr = c_str(arg)
        value[0].v_str = tstr
        tcode[0] = kTVMStr
        temp_args.append(tstr)
    elif isinstance(arg, (list, tuple, dict, _CLASS_OBJECT_GENERIC)):
        arg = _FUNC_CONVERT_TO_OBJECT(arg)
        value[0].v_handle = (<ObjectBase>arg).chandle
        tcode[0] = kTVMObjectHandle
        temp_args.append(arg)
    elif isinstance(arg, _CLASS_MODULE):
        value[0].v_handle = c_handle(arg.handle)
        tcode[0] = kTVMModuleHandle
    elif isinstance(arg, PackedFuncBase):
        value[0].v_handle = (<PackedFuncBase>arg).chandle
        tcode[0] = kTVMPackedFuncHandle
    elif isinstance(arg, ctypes.c_void_p):
        value[0].v_handle = c_handle(arg)
        tcode[0] = kTVMOpaqueHandle
    elif isinstance(arg, ObjectRValueRef):
        value[0].v_handle = &((<ObjectBase>(arg.obj)).chandle)
        tcode[0] = kTVMObjectRefArg
    elif callable(arg):
        arg = convert_to_tvm_func(arg)
        value[0].v_handle = (<PackedFuncBase>arg).chandle
        tcode[0] = kTVMPackedFuncHandle
        temp_args.append(arg)
    else:
        raise TypeError("Don't know how to handle type %s" % type(arg))
    return 0


cdef inline bytearray make_ret_bytes(void* chandle):
    handle = ctypes_handle(chandle)
    arr = ctypes.cast(handle, ctypes.POINTER(TVMByteArray))[0]
    size = arr.size
    res = bytearray(size)
    rptr = (ctypes.c_byte * size).from_buffer(res)
    if not ctypes.memmove(rptr, arr.data, size):
        raise RuntimeError('memmove failed')
    return res


cdef inline object make_ret(TVMValue value, int tcode):
    """convert result to return value."""
    if tcode == kTVMObjectHandle:
        return make_ret_object(value.v_handle)
    elif tcode == kTVMNullptr:
        return None
    elif tcode == kInt:
        return value.v_int64
    elif tcode == kFloat:
        return value.v_float64
    elif tcode == kTVMNDArrayHandle:
        return c_make_array(value.v_handle, False, True)
    elif tcode == kTVMStr:
        return py_str(value.v_str)
    elif tcode == kTVMBytes:
        return make_ret_bytes(value.v_handle)
    elif tcode == kTVMOpaqueHandle:
        return ctypes_handle(value.v_handle)
    elif tcode == kTVMContext:
        return TVMContext(value.v_ctx.device_type, value.v_ctx.device_id)
    elif tcode == kTVMModuleHandle:
        return _CLASS_MODULE(ctypes_handle(value.v_handle))
    elif tcode == kTVMPackedFuncHandle:
        return make_packed_func(value.v_handle, False)
    elif tcode in _TVM_EXT_RET:
        return _TVM_EXT_RET[tcode](ctypes_handle(value.v_handle))

    raise ValueError("Unhandled type code %d" % tcode)


cdef inline int FuncCall3(void* chandle,
                          tuple args,
                          int nargs,
                          TVMValue* ret_val,
                          int* ret_tcode) except -1:
    cdef TVMValue[3] values
    cdef int[3] tcodes
    nargs = len(args)
    temp_args = []
    for i in range(nargs):
        make_arg(args[i], &values[i], &tcodes[i], temp_args)
    CALL(TVMFuncCall(chandle, &values[0], &tcodes[0],
                     nargs, ret_val, ret_tcode))
    return 0

cdef inline int FuncCall(void* chandle,
                         tuple args,
                         TVMValue* ret_val,
                         int* ret_tcode) except -1:
    cdef int nargs
    nargs = len(args)
    if nargs <= 3:
        FuncCall3(chandle, args, nargs, ret_val, ret_tcode)
        return 0

    cdef vector[TVMValue] values
    cdef vector[int] tcodes
    values.resize(max(nargs, 1))
    tcodes.resize(max(nargs, 1))
    temp_args = []
    for i in range(nargs):
        make_arg(args[i], &values[i], &tcodes[i], temp_args)
    CALL(TVMFuncCall(chandle, &values[0], &tcodes[0],
                     nargs, ret_val, ret_tcode))
    return 0


cdef inline int ConstructorCall(void* constructor_handle,
                                int type_code,
                                tuple args,
                                void** handle) except -1:
    """Call contructor of a handle function"""
    cdef TVMValue ret_val
    cdef int ret_tcode
    FuncCall(constructor_handle, args, &ret_val, &ret_tcode)
    assert ret_tcode == type_code
    handle[0] = ret_val.v_handle
    return 0


cdef class PackedFuncBase:
    cdef TVMPackedFuncHandle chandle
    cdef int is_global

    cdef inline _set_handle(self, handle):
        if handle is None:
            self.chandle = NULL
        else:
            self.chandle = c_handle(handle)

    property is_global:
        def __get__(self):
            return self.c_is_global != 0

        def __set__(self, value):
            self.c_is_global = value

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.chandle, ctypes.c_void_p)
        def __set__(self, value):
            self._set_handle(value)

    def __init__(self, handle, is_global):
        self._set_handle(handle)
        self.c_is_global = is_global

    def __dealloc__(self):
        if self.is_global == 0:
            CALL(TVMFuncFree(self.chandle))

    def __call__(self, *args):
        cdef TVMValue ret_val
        cdef int ret_tcode
        FuncCall(self.chandle, args, &ret_val, &ret_tcode)
        return make_ret(ret_val, ret_tcode)


def _get_global_func(name, allow_missing):
    cdef TVMPackedFuncHandle chandle
    CALL(TVMFuncGetGlobal(c_str(name), &chandle))
    if chandle != NULL:
        return make_packed_func(chandle, True)

    if allow_missing:
       return None

    raise ValueError("Cannot find global function %s" % name)


_CLASS_PACKED_FUNC = None
_CLASS_MODULE = None
_CLASS_OBJECT = None
_CLASS_OBJECT_GENERIC = None
_FUNC_CONVERT_TO_OBJECT = None

def _set_class_module(module_class):
    """Initialize the module."""
    global _CLASS_MODULE
    _CLASS_MODULE = module_class

def _set_class_packed_func(func_class):
    global _CLASS_PACKED_FUNC
    _CLASS_PACKED_FUNC = func_class

def _set_class_object(obj_class):
    global _CLASS_OBJECT
    _CLASS_OBJECT = obj_class

def _set_class_object_generic(object_generic_class, func_convert_to_object):
    global _CLASS_OBJECT_GENERIC
    global _FUNC_CONVERT_TO_OBJECT
    _CLASS_OBJECT_GENERIC = object_generic_class
    _FUNC_CONVERT_TO_OBJECT = func_convert_to_object
