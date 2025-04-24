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
from numbers import Number, Integral


cdef inline object make_ret(TVMFFIAny result):
    """convert result to return value."""
   # TODO: Implement
    cdef int32_t type_index
    type_index = result.type_index
    if type_index >= kTVMFFIStaticObjectBegin:
        return make_ret_object(result)
    elif type_index == kTVMFFINone:
        return None
    elif type_index == kTVMFFIBool:
        return bool(result.v_int64)
    elif type_index == kTVMFFIInt:
        return result.v_int64
    elif type_index == kTVMFFIFloat:
        return result.v_float64
    elif type_index == kTVMFFIOpaquePtr:
        return ctypes_handle(result.v_ptr)
    elif type_index == kTVMFFIDataType:
        return make_ret_dtype(result)
    elif type_index == kTVMFFIDLTensorPtr:
        return make_ret_dltensor(result)
    elif type_index == kTVMFFIObjectRValueRef:
        raise NotImplementedError()
    elif type_index == kTVMFFIByteArrayPtr:
        raise ValueError("Return value cannot be ByteArrayPtr")
    elif type_index == kTVMFFIRawStr:
        raise ValueError("Return value cannot be RawStr")
    raise ValueError("Unhandled type index %d" % type_index)


cdef inline int make_args(tuple py_args, TVMFFIAny* out, list temp_args) except -1:
    """Pack arguments into c args tvm call accept"""
    cdef unsigned long long ptr
    for i, arg in enumerate(py_args):
        # clear the value to ensure zero padding on 32bit platforms
        if sizeof(void*) != 8:
            out[i].v_int64 = 0

        if isinstance(arg, Object):
            out[i].type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
            out[i].v_ptr = (<Object>arg).chandle
        elif isinstance(arg, PyNativeObject):
            arg = arg.__tvm_object__
            out[i].type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
            out[i].v_ptr = (<Object>arg).chandle
        elif isinstance(arg, bool):
            # A python `bool` is a subclass of `int`, so this check
            # must occur before `Integral`.
            out[i].type_index = kTVMFFIBool
            out[i].v_int64 = arg
        elif isinstance(arg, Integral):
            out[i].type_index = kTVMFFIInt
            out[i].v_int64 = arg
        elif isinstance(arg, float):
            out[i].type_index = kTVMFFIFloat
            out[i].v_float64 = arg
        elif isinstance(arg, str):
            tstr = c_str(arg)
            out[i].type_index = kTVMFFIRawStr
            out[i].v_c_str = tstr
            temp_args.append(tstr)
        elif arg is None:
            out[i].type_index = kTVMFFINone
            out[i].v_int64 = 0
        elif isinstance(arg, Number):
            out[i].type_index = kTVMFFIFloat
            out[i].v_float64 = arg
        elif isinstance(arg, _CLASS_DTYPE):
            arg = arg.__tvm_object__
            out[i].type_index = kTVMFFIDataType
            out[i].v_dtype = (<DataType>arg).cdtype
        elif isinstance(arg, (bytes, bytearray)):
            arg = ByteArrayArg(arg)
            out[i].type_index = kTVMFFIByteArrayPtr
            out[i].v_int64 = 0
            out[i].v_ptr = &((<ByteArrayArg>arg).cdata)
            temp_args.append(arg)
        elif isinstance(arg, (list, tuple, dict, _CLASS_OBJECT_GENERIC)):
            arg = _FUNC_CONVERT_TO_OBJECT(arg)
            out[i].type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
            out[i].v_ptr = (<Object>arg).chandle
            temp_args.append(arg)
        elif isinstance(arg, ctypes.c_void_p):
            out[i].type_index = kTVMFFIOpaquePtr
            out[i].v_ptr = c_handle(arg)
        elif callable(arg):
            raise NotImplementedError()


cdef inline int FuncCall3(void* chandle,
                          tuple args,
                          TVMFFIAny* result) except -1:
    # fast path with stack alloca for less than 3 args
    cdef TVMFFIAny[3] packed_args
    cdef int nargs = len(args)
    cdef int c_api_ret_code
    temp_args = []
    make_args(args, &packed_args[0], temp_args)
    with nogil:
        c_api_ret_code = TVMFFIFuncCall(
            chandle, &packed_args[0], nargs, result
        )
    CHECK_CALL(c_api_ret_code)
    return 0


cdef inline int FuncCall(void* chandle,
                         tuple args,
                         TVMFFIAny* result) except -1:
    cdef int nargs = len(args)
    cdef int c_api_ret_code

    if nargs <= 3:
        FuncCall3(chandle, args, result)
        return 0

    cdef vector[TVMFFIAny] packed_args
    packed_args.resize(nargs)

    temp_args = []
    make_args(args, &packed_args[0], temp_args)

    with nogil:
        c_api_ret_code = TVMFFIFuncCall(chandle, &packed_args[0], nargs, result)
    CHECK_CALL(c_api_ret_code)
    return 0


cdef inline int ConstructorCall(void* constructor_handle,
                                tuple args,
                                void** handle) except -1:
    """Call contructor of a handle function"""
    cdef TVMFFIAny result
    FuncCall(constructor_handle, args, &result)
    handle[0] = result.v_ptr
    return 0


cdef class Function(Object):
    """The Function object used in TVM FFI.

    See Also
    --------
    tvm.ffi.register_func: How to register global function.
    tvm.ffi.get_global_func: How to get global function.
    """
    def __call__(self, *args):
        cdef TVMFFIAny result
        FuncCall(self.chandle, args, &result)
        return make_ret(result)


def _get_global_func(name, allow_missing):
    cdef TVMFFIObjectHandle chandle

    CHECK_CALL(TVMFFIFuncGetGlobal(c_str(name), &chandle))
    if chandle != NULL:
        ret = Function.__new__(Function)
        (<Object>ret).chandle = chandle
        return ret

    if allow_missing:
       return None

    raise ValueError("Cannot find global function %s" % name)
