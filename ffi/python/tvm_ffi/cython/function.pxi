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
import os
from numbers import Real, Integral


if os.environ.get("TVM_FFI_BUILD_DOCS", "0") == "0":
    try:
        # optionally import torch and setup torch related utils
        import torch
    except ImportError:
        torch = None
else:
    torch = None


cdef int _RELEASE_GIL_BY_DEFAULT = int(
  os.environ.get("TVM_FFI_RELEASE_GIL_BY_DEFAULT", "1")
)

cdef inline object make_ret_small_str(TVMFFIAny result):
    """convert small string to return value."""
    cdef TVMFFIByteArray bytes
    bytes = TVMFFISmallBytesGetContentByteArray(&result)
    return py_str(PyBytes_FromStringAndSize(bytes.data, bytes.size))


cdef inline object make_ret_small_bytes(TVMFFIAny result):
    """convert small bytes to return value."""
    cdef TVMFFIByteArray bytes
    bytes = TVMFFISmallBytesGetContentByteArray(&result)
    return PyBytes_FromStringAndSize(bytes.data, bytes.size)


cdef inline object make_ret(TVMFFIAny result, DLPackToPyObject c_dlpack_to_pyobject = NULL):
    """convert result to return value."""
    cdef int32_t type_index
    type_index = result.type_index
    if type_index == kTVMFFITensor:
        # specially handle Tensor as it needs a special dltensor field
        return make_tensor_from_any(result, c_dlpack_to_pyobject)
    elif type_index == kTVMFFIOpaquePyObject:
        return make_ret_opaque_object(result)
    elif type_index >= kTVMFFIStaticObjectBegin:
        return make_ret_object(result)
    # the following code should be optimized to switch case
    if type_index == kTVMFFINone:
        return None
    elif type_index == kTVMFFIBool:
        return bool(result.v_int64)
    elif type_index == kTVMFFIInt:
        return result.v_int64
    elif type_index == kTVMFFIFloat:
        return result.v_float64
    elif type_index == kTVMFFISmallStr:
        return make_ret_small_str(result)
    elif type_index == kTVMFFISmallBytes:
        return make_ret_small_bytes(result)
    elif type_index == kTVMFFIOpaquePtr:
        return ctypes_handle(result.v_ptr)
    elif type_index == kTVMFFIDataType:
        return make_ret_dtype(result)
    elif type_index == kTVMFFIDevice:
        return make_ret_device(result)
    elif type_index == kTVMFFIDLTensorPtr:
        return make_ret_dltensor(result)
    elif type_index == kTVMFFIObjectRValueRef:
        raise ValueError("Return value cannot be ObjectRValueRef")
    elif type_index == kTVMFFIByteArrayPtr:
        raise ValueError("Return value cannot be ByteArrayPtr")
    elif type_index == kTVMFFIRawStr:
        raise ValueError("Return value cannot be RawStr")
    raise ValueError("Unhandled type index %d" % type_index)


##----------------------------------------------------------------------------
## Implementation of setters using same naming style as TVMFFIPyArgSetterXXX_
##----------------------------------------------------------------------------
cdef int TVMFFIPyArgSetterTensor_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* arg, TVMFFIAny* out
) except -1:
    if (<Object>arg).chandle != NULL:
        out.type_index = kTVMFFITensor
        out.v_ptr = (<Tensor>arg).chandle
    else:
        out.type_index = kTVMFFIDLTensorPtr
        out.v_ptr = (<Tensor>arg).cdltensor
    return 0


cdef int TVMFFIPyArgSetterObject_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* arg, TVMFFIAny* out
) except -1:
    out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
    out.v_ptr = (<Object>arg).chandle
    return 0


cdef int TVMFFIPyArgSetterDLPackCExporter_(
    TVMFFIPyArgSetter* this, TVMFFIPyCallContext* ctx,
    PyObject* arg, TVMFFIAny* out
) except -1:
    cdef DLManagedTensorVersioned* temp_managed_tensor
    cdef TVMFFIObjectHandle temp_chandle
    cdef TVMFFIStreamHandle env_stream = NULL

    if this.c_dlpack_to_pyobject != NULL:
        ctx.c_dlpack_to_pyobject = this.c_dlpack_to_pyobject
    if this.c_dlpack_tensor_allocator != NULL:
        ctx.c_dlpack_tensor_allocator = this.c_dlpack_tensor_allocator

    if ctx.device_id != -1:
        # already queried device, do not do it again, pass NULL to stream
        if (this.c_dlpack_from_pyobject)(arg, &temp_managed_tensor, NULL) != 0:
            return -1
    else:
        # query string on the envrionment stream
        if (this.c_dlpack_from_pyobject)(arg, &temp_managed_tensor, &env_stream) != 0:
            return -1
        # If device is not CPU, we should set the device type and id
        if temp_managed_tensor.dl_tensor.device.device_type != kDLCPU:
            ctx.stream = env_stream
            ctx.device_type = temp_managed_tensor.dl_tensor.device.device_type
            ctx.device_id = temp_managed_tensor.dl_tensor.device.device_id
    # run conversion
    if TVMFFITensorFromDLPackVersioned(temp_managed_tensor, 0, 0, &temp_chandle) != 0:
        raise BufferError("Failed to convert DLManagedTensorVersioned to ffi.Tensor")
    out.type_index = kTVMFFITensor
    out.v_ptr = temp_chandle
    TVMFFIPyPushTempFFIObject(ctx, temp_chandle)
    return 0


cdef int TorchDLPackToPyObjectFallback_(
    DLManagedTensorVersioned* dltensor, void** py_obj_out
) except -1:
    # a bit convoluted but ok as a fallback
    cdef TVMFFIObjectHandle temp_chandle
    TVMFFITensorFromDLPackVersioned(dltensor, 0, 0, &temp_chandle)
    tensor = make_tensor_from_chandle(temp_chandle)
    torch_tensor = torch.from_dlpack(tensor)
    Py_INCREF(torch_tensor)
    py_obj_out[0] = <void*>(<PyObject*>torch_tensor)
    return 0


cdef int TVMFFIPyArgSetterTorchFallback_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Current setter for torch.Tensor, go through python and not as fast as c exporter"""
    # TODO(tqchen): remove this once torch always support fast DLPack importer
    cdef object arg = <object>py_arg
    is_cuda = arg.is_cuda
    arg = from_dlpack(torch.utils.dlpack.to_dlpack(arg))
    out.type_index = kTVMFFITensor
    out.v_ptr = (<Tensor>arg).chandle
    temp_dltensor = TVMFFITensorGetDLTensorPtr((<Tensor>arg).chandle)
    ctx.c_dlpack_to_pyobject = TorchDLPackToPyObjectFallback_
    # record the stream and device for torch context
    if is_cuda and ctx.device_type != -1:
        ctx.device_type = temp_dltensor.device.device_type
        ctx.device_id = temp_dltensor.device.device_id
        # This is an API that dynamo and other uses to get the raw stream from torch
        temp_ptr = torch._C._cuda_getCurrentRawStream(temp_dltensor.device.device_id)
        ctx.stream = <TVMFFIStreamHandle>temp_ptr
    # push to temp and clear the handle
    TVMFFIPyPushTempPyObject(ctx, <PyObject*>arg)
    return 0


cdef int TVMFFIPyArgSetterDLPack_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for __dlpack__ mechanism through python, not as fast as c exporter"""
    cdef TVMFFIObjectHandle temp_chandle
    cdef object arg = <object>py_arg
    _from_dlpack_universal(arg, 0, 0, &temp_chandle)
    out.type_index = kTVMFFITensor
    out.v_ptr = temp_chandle
    # record the stream from the source framework context when possible
    temp_dltensor = TVMFFITensorGetDLTensorPtr(temp_chandle)
    if (temp_dltensor.device.device_type != kDLCPU and
        ctx.device_type != -1):
        # __tvm_ffi_env_stream__ returns the expected stream that should be set
        # through TVMFFIEnvSetStream when calling a TVM FFI function
        if hasattr(arg, "__tvm_ffi_env_stream__"):
            # Ideally projects should directly setup their stream context API
            # write through by also calling TVMFFIEnvSetStream
            # so we do not need this protocol to do exchange
            ctx.device_type = temp_dltensor.device.device_type
            ctx.device_id = temp_dltensor.device.device_id
            temp_ptr= arg.__tvm_ffi_env_stream__()
            ctx.stream = <TVMFFIStreamHandle>temp_ptr
    TVMFFIPyPushTempFFIObject(ctx, temp_chandle)
    return 0


cdef int TVMFFIPyArgSetterDType_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for dtype"""
    cdef object arg = <object>py_arg
    # dtype is a subclass of str, so this check occur before str
    arg = arg.__tvm_ffi_dtype__
    out.type_index = kTVMFFIDataType
    out.v_dtype = (<DataType>arg).cdtype
    return 0


cdef int TVMFFIPyArgSetterDevice_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for device"""
    cdef object arg = <object>py_arg
    out.type_index = kTVMFFIDevice
    out.v_device = (<Device>arg).cdevice
    return 0


cdef int TVMFFIPyArgSetterStr_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for str"""
    cdef object arg = <object>py_arg

    if isinstance(arg, PyNativeObject) and arg.__tvm_ffi_object__ is not None:
        arg = arg.__tvm_ffi_object__
        out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
        out.v_ptr = (<Object>arg).chandle
        return 0

    tstr = c_str(arg)
    out.type_index = kTVMFFIRawStr
    out.v_c_str = tstr
    TVMFFIPyPushTempPyObject(ctx, <PyObject*>tstr)
    return 0


cdef int TVMFFIPyArgSetterBytes_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for bytes"""
    cdef object arg = <object>py_arg

    if isinstance(arg, PyNativeObject) and arg.__tvm_ffi_object__ is not None:
        arg = arg.__tvm_ffi_object__
        out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
        out.v_ptr = (<Object>arg).chandle
        return 0

    arg = ByteArrayArg(arg)
    out.type_index = kTVMFFIByteArrayPtr
    out.v_int64 = 0
    out.v_ptr = (<ByteArrayArg>arg).cptr()
    TVMFFIPyPushTempPyObject(ctx, <PyObject*>arg)
    return 0


cdef int TVMFFIPyArgSetterCtypesVoidPtr_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for ctypes.c_void_p"""
    out.type_index = kTVMFFIOpaquePtr
    out.v_ptr = c_handle(<object>py_arg)
    return 0


cdef int TVMFFIPyArgSetterObjectRValueRef_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for ObjectRValueRef"""
    cdef object arg = <object>py_arg
    out.type_index = kTVMFFIObjectRValueRef
    out.v_ptr = &((<Object>(arg.obj)).chandle)
    return 0


cdef int TVMFFIPyArgSetterCallable_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for Callable"""
    cdef object arg = <object>py_arg
    arg = _convert_to_ffi_func(arg)
    out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
    out.v_ptr = (<Object>arg).chandle
    TVMFFIPyPushTempPyObject(ctx, <PyObject*>arg)
    return 0


cdef int TVMFFIPyArgSetterException_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for Exception"""
    cdef object arg = <object>py_arg
    arg = _convert_to_ffi_error(arg)
    out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
    out.v_ptr = (<Object>arg).chandle
    TVMFFIPyPushTempPyObject(ctx, <PyObject*>arg)
    return 0


cdef int TVMFFIPyArgSetterFallback_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Fallback setter for all other types"""
    cdef object arg = <object>py_arg
    # fallback must contain PyNativeObject check
    if isinstance(arg, PyNativeObject) and arg.__tvm_ffi_object__ is not None:
        arg = arg.__tvm_ffi_object__
        out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
        out.v_ptr = (<Object>arg).chandle
    elif isinstance(arg, (list, tuple, dict, ObjectConvertible)):
        arg = _FUNC_CONVERT_TO_OBJECT(arg)
        out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
        out.v_ptr = (<Object>arg).chandle
        TVMFFIPyPushTempPyObject(ctx, <PyObject*>arg)
    else:
        arg = _convert_to_opaque_object(arg)
        out.type_index = kTVMFFIOpaquePyObject
        out.v_ptr = (<Object>arg).chandle
        TVMFFIPyPushTempPyObject(ctx, <PyObject*>arg)


cdef int TVMFFIPyArgSetterFactory_(PyObject* value, TVMFFIPyArgSetter* out) except -1:
    """
    Factory function that creates an argument setter for a given Python argument type.
    """
    # NOTE: the order of checks matter here
    # becase each argument may satisfy multiple checks
    # priortize native types over external types
    cdef object arg = <object>value
    cdef long long temp_ptr
    if arg is None:
        out.func = TVMFFIPyArgSetterNone_
        return 0
    if isinstance(arg, Tensor):
        out.func = TVMFFIPyArgSetterTensor_
        return 0
    if isinstance(arg, Object):
        out.func = TVMFFIPyArgSetterObject_
        return 0
    if isinstance(arg, ObjectRValueRef):
        out.func = TVMFFIPyArgSetterObjectRValueRef_
        return 0
    if os.environ.get("TVM_FFI_SKIP_c_dlpack_from_pyobject", "0") != "1":
        # external tensors
        if hasattr(arg, "__c_dlpack_from_pyobject__"):
            out.func = TVMFFIPyArgSetterDLPackCExporter_
            temp_ptr = arg.__c_dlpack_from_pyobject__
            out.c_dlpack_from_pyobject = <DLPackFromPyObject>temp_ptr
            if hasattr(arg, "__c_dlpack_to_pyobject__"):
                temp_ptr = arg.__c_dlpack_to_pyobject__
                out.c_dlpack_to_pyobject = <DLPackToPyObject>temp_ptr
            if hasattr(arg, "__c_dlpack_tensor_allocator__"):
                temp_ptr = arg.__c_dlpack_tensor_allocator__
                out.c_dlpack_tensor_allocator = <DLPackTensorAllocator>temp_ptr
            return 0
    if torch is not None and isinstance(arg, torch.Tensor):
        out.func = TVMFFIPyArgSetterTorchFallback_
        return 0
    if hasattr(arg, "__dlpack__"):
        out.func = TVMFFIPyArgSetterDLPack_
        return 0
    if isinstance(arg, bool):
        # A python `bool` is a subclass of `int`, so this check
        # must occur before `Integral`.
        out.func = TVMFFIPyArgSetterBool_
        return 0
    if isinstance(arg, Integral):
        out.func = TVMFFIPyArgSetterInt_
        return 0
    if isinstance(arg, Real):
        out.func = TVMFFIPyArgSetterFloat_
        return 0
    # dtype is a subclass of str, so this check must occur before str
    if isinstance(arg, _CLASS_DTYPE):
        out.func = TVMFFIPyArgSetterDType_
        return 0
    if isinstance(arg, _CLASS_DEVICE):
        out.func = TVMFFIPyArgSetterDevice_
        return 0
    if isinstance(arg, str):
        out.func = TVMFFIPyArgSetterStr_
        return 0
    if isinstance(arg, (bytes, bytearray)):
        out.func = TVMFFIPyArgSetterBytes_
        return 0
    if isinstance(arg, ctypes.c_void_p):
        out.func = TVMFFIPyArgSetterCtypesVoidPtr_
        return 0
    if callable(arg):
        out.func = TVMFFIPyArgSetterCallable_
        return 0
    if isinstance(arg, Exception):
        out.func = TVMFFIPyArgSetterException_
        return 0
    # default to opaque object
    out.func = TVMFFIPyArgSetterFallback_
    return 0

#---------------------------------------------------------------------------------------------
## Implementation of function calling
#---------------------------------------------------------------------------------------------
cdef inline int ConstructorCall(void* constructor_handle,
                                tuple args,
                                void** handle) except -1:
    """Call contructor of a handle function"""
    cdef TVMFFIAny result
    cdef int c_api_ret_code
    # IMPORTANT: caller need to initialize result->type_index to kTVMFFINone
    result.type_index = kTVMFFINone
    result.v_int64 = 0
    TVMFFIPyFuncCall(
        TVMFFIPyArgSetterFactory_, constructor_handle, <PyObject*>args, &result, &c_api_ret_code,
        False, NULL
    )
    CHECK_CALL(c_api_ret_code)
    handle[0] = result.v_ptr
    return 0


cdef class Function(Object):
    """Python class that wraps a function with tvm-ffi ABI.

    See Also
    --------
    tvm_ffi.register_global_func: How to register global function.
    tvm_ffi.get_global_func: How to get global function.
    """
    cdef int c_release_gil
    cdef dict __dict__

    def __cinit__(self):
        self.c_release_gil = _RELEASE_GIL_BY_DEFAULT

    property release_gil:
        def __get__(self):
            return self.c_release_gil != 0
        def __set__(self, value):
            self.c_release_gil = value

    def __call__(self, *args):
        cdef TVMFFIAny result
        cdef int c_api_ret_code
        cdef DLPackToPyObject c_dlpack_to_pyobject = NULL
        # IMPORTANT: caller need to initialize result->type_index to kTVMFFINone
        result.type_index = kTVMFFINone
        result.v_int64 = 0
        TVMFFIPyFuncCall(
            TVMFFIPyArgSetterFactory_,
            (<Object>self).chandle, <PyObject*>args,
            &result,
            &c_api_ret_code,
            self.release_gil,
            &c_dlpack_to_pyobject
        )
        # NOTE: logic is same as check_call
        # directly inline here to simplify traceback
        if c_api_ret_code == 0:
            return make_ret(result, c_dlpack_to_pyobject)
        elif c_api_ret_code == -2:
            raise_existing_error()
        raise move_from_last_error().py_error()

_register_object_by_index(kTVMFFIFunction, Function)


cdef class FieldGetter:
    cdef TVMFFIFieldGetter getter
    cdef int64_t offset

    def __call__(self, Object obj):
        cdef TVMFFIAny result
        cdef int c_api_ret_code
        cdef void* field_ptr = (<char*>(<Object>obj).chandle) + self.offset
        result.type_index = kTVMFFINone
        result.v_int64 = 0
        c_api_ret_code = self.getter(field_ptr, &result)
        CHECK_CALL(c_api_ret_code)
        return make_ret(result)


cdef class FieldSetter:
    cdef TVMFFIFieldSetter setter
    cdef int64_t offset

    def __call__(self, Object obj, value):
        cdef int c_api_ret_code
        cdef void* field_ptr = (<char*>(<Object>obj).chandle) + self.offset
        TVMFFIPyCallFieldSetter(
            TVMFFIPyArgSetterFactory_,
            self.setter,
            field_ptr,
            <PyObject*>value,
            &c_api_ret_code
        )
        # NOTE: logic is same as check_call
        # directly inline here to simplify traceback
        if c_api_ret_code == 0:
            return
        elif c_api_ret_code == -2:
            raise_existing_error()
        raise move_from_last_error().py_error()


cdef _get_method_from_method_info(const TVMFFIMethodInfo* method):
    cdef TVMFFIAny result
    CHECK_CALL(TVMFFIAnyViewToOwnedAny(&(method.method), &result))
    return make_ret(result)


def _member_method_wrapper(method_func):
    def wrapper(self, *args):
        return method_func(self, *args)
    return wrapper


def _add_class_attrs_by_reflection(int type_index, object cls):
    """Decorate the class attrs by reflection"""
    cdef const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(type_index)
    cdef const TVMFFIFieldInfo* field
    cdef const TVMFFIMethodInfo* method
    cdef int num_fields = info.num_fields
    cdef int num_methods = info.num_methods

    for i in range(num_fields):
        # attach fields to the class
        field = &(info.fields[i])
        getter = FieldGetter.__new__(FieldGetter)
        (<FieldGetter>getter).getter = field.getter
        (<FieldGetter>getter).offset = field.offset
        setter = FieldSetter.__new__(FieldSetter)
        (<FieldSetter>setter).setter = field.setter
        (<FieldSetter>setter).offset = field.offset
        if (field.flags & kTVMFFIFieldFlagBitMaskWritable) == 0:
            setter = None
        doc = (
            py_str(PyBytes_FromStringAndSize(field.doc.data, field.doc.size))
            if field.doc.size != 0
            else None
        )
        name = py_str(PyBytes_FromStringAndSize(field.name.data, field.name.size))
        if hasattr(cls, name):
            # skip already defined attributes
            continue
        setattr(cls, name, property(getter, setter, doc=doc))

    for i in range(num_methods):
        # attach methods to the class
        method = &(info.methods[i])
        name = py_str(PyBytes_FromStringAndSize(method.name.data, method.name.size))
        doc = (
            py_str(PyBytes_FromStringAndSize(method.doc.data, method.doc.size))
            if method.doc.size != 0
            else None
        )
        method_func = _get_method_from_method_info(method)

        if method.flags & kTVMFFIFieldFlagBitMaskIsStaticMethod:
            method_pyfunc = staticmethod(method_func)
        else:
            # must call into another method instead of direct capture
            # to avoid the same method_func variable being used
            # across multiple loop iterations
            method_pyfunc = _member_method_wrapper(method_func)

        if doc is not None:
            method_pyfunc.__doc__ = doc
            method_pyfunc.__name__ = name

        if hasattr(cls, name):
            # skip already defined attributes
            continue
        setattr(cls, name, method_pyfunc)

    return cls


def _register_global_func(name, pyfunc, override):
    cdef TVMFFIObjectHandle chandle
    cdef int c_api_ret_code
    cdef int ioverride = override
    cdef ByteArrayArg name_arg = ByteArrayArg(c_str(name))

    if not isinstance(pyfunc, Function):
        pyfunc = _convert_to_ffi_func(pyfunc)

    CHECK_CALL(TVMFFIFunctionSetGlobal(name_arg.cptr(), (<Object>pyfunc).chandle, ioverride))
    return pyfunc


def _get_global_func(name, allow_missing):
    cdef TVMFFIObjectHandle chandle
    cdef ByteArrayArg name_arg = ByteArrayArg(c_str(name))

    CHECK_CALL(TVMFFIFunctionGetGlobal(name_arg.cptr(), &chandle))
    if chandle != NULL:
        ret = Function.__new__(Function)
        (<Object>ret).chandle = chandle
        return ret

    if allow_missing:
       return None

    raise ValueError("Cannot find global function %s" % name)


# handle callbacks
cdef void tvm_ffi_pyobject_deleter(void* fhandle) noexcept with gil:
    local_pyobject = <object>(fhandle)
    Py_DECREF(local_pyobject)


cdef int tvm_ffi_callback(void* context,
                          const TVMFFIAny* packed_args,
                          int32_t num_args,
                          TVMFFIAny* result) noexcept with gil:
    cdef list pyargs
    cdef TVMFFIAny temp_result
    cdef int c_api_ret_code
    local_pyfunc = <object>(context)
    pyargs = []
    for i in range(num_args):
        CHECK_CALL(TVMFFIAnyViewToOwnedAny(&packed_args[i], &temp_result))
        pyargs.append(make_ret(temp_result))

    try:
        rv = local_pyfunc(*pyargs)
        TVMFFIPyPyObjectToFFIAny(
            TVMFFIPyArgSetterFactory_,
            <PyObject*>rv,
            result,
            &c_api_ret_code
        )
        if c_api_ret_code == 0:
            return 0
        elif c_api_ret_code == -2:
            raise_existing_error()
        return -1
    except Exception as err:
        set_last_ffi_error(err)
        return -1


def _convert_to_ffi_func(object pyfunc):
    """Convert a python function to TVM FFI function"""
    cdef TVMFFIObjectHandle chandle
    Py_INCREF(pyfunc)
    CHECK_CALL(TVMFFIFunctionCreate(
        <void*>(pyfunc),
        tvm_ffi_callback,
        tvm_ffi_pyobject_deleter,
        &chandle))
    ret = Function.__new__(Function)
    (<Object>ret).chandle = chandle
    return ret


def _convert_to_opaque_object(object pyobject):
    """Convert a python object to TVM FFI opaque object"""
    cdef TVMFFIObjectHandle chandle
    Py_INCREF(pyobject)
    CHECK_CALL(TVMFFIObjectCreateOpaque(
        <void*>(pyobject),
        kTVMFFIOpaquePyObject,
        tvm_ffi_pyobject_deleter,
        &chandle))
    ret = OpaquePyObject.__new__(OpaquePyObject)
    (<Object>ret).chandle = chandle
    return ret


def _print_debug_info():
    """Get the size of the dispatch map"""
    cdef size_t size =   TVMFFIPyGetDispatchMapSize()
    print(f"TVMFFIPyGetDispatchMapSize: {size}")


_STR_CONSTRUCTOR = _get_global_func("ffi.String", False)
_BYTES_CONSTRUCTOR = _get_global_func("ffi.Bytes", False)
_OBJECT_FROM_JSON_GRAPH_STR = _get_global_func("ffi.FromJSONGraphString", True)
_OBJECT_TO_JSON_GRAPH_STR = _get_global_func("ffi.ToJSONGraphString", True)
