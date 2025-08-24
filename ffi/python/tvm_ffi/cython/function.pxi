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
from numbers import Real, Integral

try:
    # optionally import torch and setup torch related utils
    import torch
except ImportError:
    torch = None


def load_torch_get_current_cuda_stream():
    """Create a faster get_current_cuda_stream for torch through cpp extension.
    """
    from torch.utils import cpp_extension

    source = """
    #include <c10/cuda/CUDAStream.h>

    int64_t get_current_cuda_stream(int device_id) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device_id);
        // fast invariant, default stream is always 0
        if (stream.id() == 0) return 0;
        // convert to cudaStream_t
        return reinterpret_cast<int64_t>(static_cast<cudaStream_t>(stream));
    }
    """
    def fallback_get_current_cuda_stream(device_id):
        """Fallback with python api"""
        return torch.cuda.current_stream(device_id).cuda_stream
    try:
        result = cpp_extension.load_inline(
            name="get_current_cuda_stream",
            cpp_sources=[source],
            cuda_sources=[],
            extra_cflags=["-O3"],
            extra_include_paths=cpp_extension.include_paths("cuda"),
            functions=["get_current_cuda_stream"],
        )
        return result.get_current_cuda_stream
    except Exception:
        return fallback_get_current_cuda_stream

if torch is not None:
    # when torch is available, jit compile the get_current_cuda_stream function
    # the torch caches the extension so second loading is faster
    torch_get_current_cuda_stream = load_torch_get_current_cuda_stream()


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


cdef inline object make_ret(TVMFFIAny result):
    """convert result to return value."""
   # TODO: Implement
    cdef int32_t type_index
    type_index = result.type_index
    if type_index == kTVMFFINDArray:
        # specially handle NDArray as it needs a special dltensor field
        return make_ndarray_from_any(result)
    elif type_index >= kTVMFFIStaticObjectBegin:
        return make_ret_object(result)
    elif type_index == kTVMFFINone:
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


cdef inline int make_args(tuple py_args, TVMFFIAny* out, list temp_args,
                          int* ctx_dev_type, int* ctx_dev_id, TVMFFIStreamHandle* ctx_stream) except -1:
    """Pack arguments into c args tvm call accept"""
    cdef unsigned long long temp_ptr
    cdef DLTensor* temp_dltensor
    cdef int is_cuda = 0

    for i, arg in enumerate(py_args):
        # clear the value to ensure zero padding on 32bit platforms
        if sizeof(void*) != 8:
            out[i].v_int64 = 0
        out[i].zero_padding = 0

        if isinstance(arg, NDArray):
            if (<Object>arg).chandle != NULL:
                out[i].type_index = kTVMFFINDArray
                out[i].v_ptr = (<NDArray>arg).chandle
            else:
                out[i].type_index = kTVMFFIDLTensorPtr
                out[i].v_ptr = (<NDArray>arg).cdltensor
        elif isinstance(arg, Object):
            out[i].type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
            out[i].v_ptr = (<Object>arg).chandle
        elif torch is not None and isinstance(arg, torch.Tensor):
            is_cuda = arg.is_cuda
            arg = from_dlpack(torch.utils.dlpack.to_dlpack(arg),
                              required_alignment=__dlpack_auto_import_required_alignment__)
            out[i].type_index = kTVMFFINDArray
            out[i].v_ptr = (<NDArray>arg).chandle
            temp_dltensor = TVMFFINDArrayGetDLTensorPtr((<NDArray>arg).chandle)
            # record the stream and device for torch context
            if is_cuda and ctx_dev_type != NULL and ctx_dev_type[0] == -1:
                ctx_dev_type[0] = temp_dltensor.device.device_type
                ctx_dev_id[0] = temp_dltensor.device.device_id
                temp_ptr = torch_get_current_cuda_stream(temp_dltensor.device.device_id)
                ctx_stream[0] = <TVMFFIStreamHandle>temp_ptr
            temp_args.append(arg)
        elif hasattr(arg, "__dlpack__"):
            arg = from_dlpack(arg, required_alignment=__dlpack_auto_import_required_alignment__)
            out[i].type_index = kTVMFFINDArray
            out[i].v_ptr = (<NDArray>arg).chandle
            temp_args.append(arg)
        elif isinstance(arg, PyNativeObject) and arg.__tvm_ffi_object__ is not None:
            arg = arg.__tvm_ffi_object__
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
        elif isinstance(arg, _CLASS_DTYPE):
            # dtype is a subclass of str, so this check occur before str
            arg = arg.__tvm_ffi_dtype__
            out[i].type_index = kTVMFFIDataType
            out[i].v_dtype = (<DataType>arg).cdtype
        elif isinstance(arg, _CLASS_DEVICE):
            out[i].type_index = kTVMFFIDevice
            out[i].v_device = (<Device>arg).cdevice
        elif isinstance(arg, str):
            tstr = c_str(arg)
            out[i].type_index = kTVMFFIRawStr
            out[i].v_c_str = tstr
            temp_args.append(tstr)
        elif arg is None:
            out[i].type_index = kTVMFFINone
        elif isinstance(arg, Real):
            out[i].type_index = kTVMFFIFloat
            out[i].v_float64 = arg
        elif isinstance(arg, (bytes, bytearray)):
            arg = ByteArrayArg(arg)
            out[i].type_index = kTVMFFIByteArrayPtr
            out[i].v_int64 = 0
            out[i].v_ptr = (<ByteArrayArg>arg).cptr()
            temp_args.append(arg)
        elif isinstance(arg, (list, tuple, dict, ObjectGeneric)):
            arg = _FUNC_CONVERT_TO_OBJECT(arg)
            out[i].type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
            out[i].v_ptr = (<Object>arg).chandle
            temp_args.append(arg)
        elif isinstance(arg, ctypes.c_void_p):
            out[i].type_index = kTVMFFIOpaquePtr
            out[i].v_ptr = c_handle(arg)
        elif isinstance(arg, Exception):
            arg = _convert_to_ffi_error(arg)
            out[i].type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
            out[i].v_ptr = (<Object>arg).chandle
            temp_args.append(arg)
        elif isinstance(arg, ObjectRValueRef):
            out[i].type_index = kTVMFFIObjectRValueRef
            out[i].v_ptr = &((<Object>(arg.obj)).chandle)
        elif callable(arg):
            arg = _convert_to_ffi_func(arg)
            out[i].type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
            out[i].v_ptr = (<Object>arg).chandle
            temp_args.append(arg)
        else:
            raise TypeError("Unsupported argument type: %s" % type(arg))


cdef inline int FuncCall3(void* chandle,
                          tuple args,
                          TVMFFIAny* result,
                          int* c_api_ret_code) except -1:
    # fast path with stack alloca for less than 3 args
    cdef TVMFFIAny[3] packed_args
    cdef int nargs = len(args)
    cdef int ctx_dev_type = -1
    cdef int ctx_dev_id = 0
    cdef TVMFFIStreamHandle ctx_stream = NULL
    cdef TVMFFIStreamHandle prev_stream = NULL
    temp_args = []
    make_args(args, &packed_args[0], temp_args, &ctx_dev_type, &ctx_dev_id, &ctx_stream)
    with nogil:
        if ctx_dev_type != -1:
            # set the stream based on ctx stream
            c_api_ret_code[0] = TVMFFIEnvSetStream(ctx_dev_type, ctx_dev_id, ctx_stream, &prev_stream)
            if c_api_ret_code[0] != 0:
                return 0
        c_api_ret_code[0] = TVMFFIFunctionCall(
            chandle, &packed_args[0], nargs, result
        )
        # restore the original stream if it is not the same as the context stream
        if ctx_dev_type != -1 and prev_stream != ctx_stream:
            # restore the original stream
            c_api_ret_code[0] = TVMFFIEnvSetStream(ctx_dev_type, ctx_dev_id, prev_stream, NULL)
            if c_api_ret_code[0] != 0:
                return 0
    return 0


cdef inline int FuncCall(void* chandle,
                         tuple args,
                         TVMFFIAny* result,
                         int* c_api_ret_code) except -1:
    cdef int nargs = len(args)
    cdef int ctx_dev_type = -1
    cdef int ctx_dev_id = 0
    cdef TVMFFIStreamHandle ctx_stream = NULL
    cdef TVMFFIStreamHandle prev_stream = NULL

    if nargs <= 3:
        FuncCall3(chandle, args, result, c_api_ret_code)
        return 0

    cdef vector[TVMFFIAny] packed_args
    packed_args.resize(nargs)

    temp_args = []
    make_args(args, &packed_args[0], temp_args, &ctx_dev_type, &ctx_dev_id, &ctx_stream)

    with nogil:
        if ctx_dev_type != -1:
            c_api_ret_code[0] = TVMFFIEnvSetStream(ctx_dev_type, ctx_dev_id, ctx_stream, &prev_stream)
            if c_api_ret_code[0] != 0:
                return 0
        c_api_ret_code[0] = TVMFFIFunctionCall(chandle, &packed_args[0], nargs, result)
        # restore the original stream if it is not the same as the context stream
        if ctx_dev_type != -1 and prev_stream != ctx_stream:
            c_api_ret_code[0] = TVMFFIEnvSetStream(ctx_dev_type, ctx_dev_id, prev_stream, NULL)
            if c_api_ret_code[0] != 0:
                return 0

    return 0


cdef inline int ConstructorCall(void* constructor_handle,
                                tuple args,
                                void** handle) except -1:
    """Call contructor of a handle function"""
    cdef TVMFFIAny result
    cdef int c_api_ret_code
    # IMPORTANT: caller need to initialize result->type_index to kTVMFFINone
    result.type_index = kTVMFFINone
    result.v_int64 = 0
    FuncCall(constructor_handle, args, &result, &c_api_ret_code)
    CHECK_CALL(c_api_ret_code)
    handle[0] = result.v_ptr
    return 0


class Function(Object):
    """The Function object used in TVM FFI.

    See Also
    --------
    tvm_ffi.register_func: How to register global function.
    tvm_ffi.get_global_func: How to get global function.
    """
    def __call__(self, *args):
        cdef TVMFFIAny result
        cdef int c_api_ret_code
        # IMPORTANT: caller need to initialize result->type_index to kTVMFFINone
        result.type_index = kTVMFFINone
        result.v_int64 = 0
        FuncCall((<Object>self).chandle, args, &result, &c_api_ret_code)
        # NOTE: logic is same as check_call
        # directly inline here to simplify traceback
        if c_api_ret_code == 0:
            return make_ret(result)
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
        cdef TVMFFIAny[1] packed_args
        cdef int c_api_ret_code
        cdef void* field_ptr = (<char*>(<Object>obj).chandle) + self.offset
        cdef int nargs = 1
        temp_args = []
        make_args((value,), &packed_args[0], temp_args, NULL, NULL, NULL)
        c_api_ret_code = self.setter(field_ptr, &packed_args[0])
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
cdef void tvm_ffi_callback_deleter(void* fhandle) noexcept with gil:
    local_pyfunc = <object>(fhandle)
    Py_DECREF(local_pyfunc)


cdef int tvm_ffi_callback(void* context,
                          const TVMFFIAny* packed_args,
                          int32_t num_args,
                          TVMFFIAny* result) noexcept with gil:
    cdef list pyargs
    cdef TVMFFIAny temp_result
    local_pyfunc = <object>(context)
    pyargs = []
    for i in range(num_args):
        CHECK_CALL(TVMFFIAnyViewToOwnedAny(&packed_args[i], &temp_result))
        pyargs.append(make_ret(temp_result))

    try:
        rv = local_pyfunc(*pyargs)
    except Exception as err:
        set_last_ffi_error(err)
        return -1

    temp_args = []
    make_args((rv,), &temp_result, temp_args, NULL, NULL, NULL)
    CHECK_CALL(TVMFFIAnyViewToOwnedAny(&temp_result, result))

    return 0


def _convert_to_ffi_func(object pyfunc):
    """Convert a python function to TVM FFI function"""
    cdef TVMFFIObjectHandle chandle
    Py_INCREF(pyfunc)
    CHECK_CALL(TVMFFIFunctionCreate(
        <void*>(pyfunc),
        tvm_ffi_callback,
        tvm_ffi_callback_deleter,
        &chandle))
    ret = Function.__new__(Function)
    (<Object>ret).chandle = chandle
    return ret

_STR_CONSTRUCTOR = _get_global_func("ffi.String", False)
_BYTES_CONSTRUCTOR = _get_global_func("ffi.Bytes", False)
_OBJECT_FROM_JSON_GRAPH_STR = _get_global_func("ffi.FromJSONGraphString", True)
_OBJECT_TO_JSON_GRAPH_STR = _get_global_func("ffi.ToJSONGraphString", True)
