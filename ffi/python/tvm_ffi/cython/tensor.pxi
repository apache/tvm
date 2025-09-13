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

__dlpack_version__ = (1, 1)
_CLASS_TENSOR = None


def _set_class_tensor(cls):
    global _CLASS_TENSOR
    _CLASS_TENSOR = cls


cdef const char* _c_str_dltensor = "dltensor"
cdef const char* _c_str_used_dltensor = "used_dltensor"
cdef const char* _c_str_dltensor_versioned = "dltensor_versioned"
cdef const char* _c_str_used_dltensor_versioned = "used_dltensor_versioned"

cdef void _c_dlpack_deleter(object pycaps):
    cdef DLManagedTensor* dltensor
    if pycapsule.PyCapsule_IsValid(pycaps, _c_str_dltensor):
        dltensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(pycaps, _c_str_dltensor)
        dltensor.deleter(dltensor)

cdef void _c_dlpack_versioned_deleter(object pycaps):
    cdef DLManagedTensorVersioned* dltensor
    if pycapsule.PyCapsule_IsValid(pycaps, _c_str_dltensor_versioned):
        dltensor = <DLManagedTensorVersioned*>pycapsule.PyCapsule_GetPointer(
            pycaps, _c_str_dltensor_versioned)
        dltensor.deleter(dltensor)


cdef inline object _from_dlpack_intptr(
    void* dlpack
):
    cdef TVMFFIObjectHandle chandle
    cdef DLManagedTensor* ptr = <DLManagedTensor*>dlpack
    cdef int c_api_ret_code
    cdef int c_req_alignment = 0
    cdef int c_req_contiguous = 0
    c_api_ret_code = TVMFFITensorFromDLPack(
        ptr, c_req_alignment, c_req_contiguous, &chandle)
    CHECK_CALL(c_api_ret_code)
    return make_tensor_from_chandle(chandle)


cdef inline int _from_dlpack(
    object dltensor, int require_alignment,
    int require_contiguous, TVMFFIObjectHandle* out
) except -1:
    cdef DLManagedTensor* ptr
    cdef int c_api_ret_code
    cdef int c_req_alignment = require_alignment
    cdef int c_req_contiguous = require_contiguous
    if pycapsule.PyCapsule_IsValid(dltensor, _c_str_dltensor):
        ptr = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(dltensor, _c_str_dltensor)
        c_api_ret_code = TVMFFITensorFromDLPack(
            ptr, c_req_alignment, c_req_contiguous, out)
        CHECK_CALL(c_api_ret_code)
        # set name and destructor to be empty
        pycapsule.PyCapsule_SetDestructor(dltensor, NULL)
        pycapsule.PyCapsule_SetName(dltensor, _c_str_used_dltensor)
        return 0
    raise ValueError("Expect a dltensor field, PyCapsule can only be consumed once")


cdef inline int _from_dlpack_versioned(
    object dltensor, int require_alignment,
    int require_contiguous, TVMFFIObjectHandle* out
) except -1:
    cdef DLManagedTensorVersioned* ptr
    cdef int c_api_ret_code
    cdef int c_req_alignment = require_alignment
    cdef int c_req_contiguous = require_contiguous
    if pycapsule.PyCapsule_IsValid(dltensor, _c_str_dltensor_versioned):
        ptr = <DLManagedTensorVersioned*>pycapsule.PyCapsule_GetPointer(
            dltensor, _c_str_dltensor_versioned)
        c_api_ret_code = TVMFFITensorFromDLPackVersioned(
            ptr, c_req_alignment, c_req_contiguous, out)
        CHECK_CALL(c_api_ret_code)
        # set name and destructor to be empty
        pycapsule.PyCapsule_SetDestructor(dltensor, NULL)
        pycapsule.PyCapsule_SetName(dltensor, _c_str_used_dltensor_versioned)
        return 0
    raise ValueError("Expect a dltensor_versioned field, PyCapsule can only be consumed once")


cdef inline int _from_dlpack_universal(
    object ext_tensor, int require_alignment,
    int require_contiguous, TVMFFIObjectHandle* out
) except -1:
    # as of most frameworks do not yet support v1.1
    # move to false as most frameworks get upgraded.
    cdef int favor_legacy_dlpack = True

    if hasattr(ext_tensor, '__dlpack__'):
        if favor_legacy_dlpack:
            _from_dlpack(
                ext_tensor.__dlpack__(),
                require_alignment,
                require_contiguous,
                out
            )
        else:
            try:
                _from_dlpack_versioned(
                    ext_tensor.__dlpack__(max_version=__dlpack_version__),
                    require_alignment,
                    require_contiguous,
                    out
                )
            except TypeError:
                _from_dlpack(
                    ext_tensor.__dlpack__(),
                    require_alignment,
                    require_contiguous,
                    out
                )
    else:
        if pycapsule.PyCapsule_IsValid(ext_tensor, _c_str_dltensor_versioned):
            _from_dlpack_versioned(
                ext_tensor,
                require_alignment,
                require_contiguous,
                out
            )
        elif pycapsule.PyCapsule_IsValid(ext_tensor, _c_str_dltensor):
            _from_dlpack(
                ext_tensor,
                require_alignment,
                require_contiguous,
                out
            )
        else:
            raise TypeError("Expect from_dlpack to take either a compatible tensor or PyCapsule")


def from_dlpack(ext_tensor, *, require_alignment=0, require_contiguous=False):
    """
    Convert an external tensor to an Tensor.

    Parameters
    ----------
    ext_tensor : object
        The external tensor to convert.

    require_alignment : int
        The minimum required alignment to check for the tensor.

    require_contiguous : bool
        Whether to check for contiguous memory.

    Returns
    -------
    tensor : :py:class:`tvm_ffi.Tensor`
        The converted tensor.
    """
    cdef TVMFFIObjectHandle chandle
    _from_dlpack_universal(ext_tensor, require_alignment, require_contiguous, &chandle)
    return make_tensor_from_chandle(chandle)


# helper class for shape handling
def _shape_obj_get_py_tuple(obj):
    cdef TVMFFIShapeCell* shape = TVMFFIShapeGetCellPtr((<Object>obj).chandle)
    return tuple(shape.data[i] for i in range(shape.size))


cdef class Tensor(Object):
    """Tensor object that represents a managed n-dimensional array.
    """
    cdef DLTensor* cdltensor

    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.cdltensor.shape[i] for i in range(self.cdltensor.ndim))

    @property
    def dtype(self):
        """Data type of this array"""
        cdef TVMFFIAny dtype_any
        dtype_any.v_dtype = self.cdltensor.dtype
        return make_ret_dtype(dtype_any)

    @property
    def device(self):
        """Device of this Tensor"""
        cdef TVMFFIAny device_any
        device_any.v_device = self.cdltensor.device
        return make_ret_device(device_any)

    def _to_dlpack(self):
        cdef DLManagedTensor* dltensor
        cdef int c_api_ret_code
        c_api_ret_code = TVMFFITensorToDLPack(self.chandle, &dltensor)
        CHECK_CALL(c_api_ret_code)
        return pycapsule.PyCapsule_New(dltensor, _c_str_dltensor, <PyCapsule_Destructor>_c_dlpack_deleter)

    def _to_dlpack_versioned(self):
        cdef DLManagedTensorVersioned* dltensor
        cdef int c_api_ret_code
        c_api_ret_code = TVMFFITensorToDLPackVersioned(self.chandle, &dltensor)
        CHECK_CALL(c_api_ret_code)
        return pycapsule.PyCapsule_New(
            dltensor, _c_str_dltensor_versioned, <PyCapsule_Destructor>_c_dlpack_versioned_deleter)

    def __dlpack_device__(self):
        cdef int device_type = self.cdltensor.device.device_type
        cdef int device_id = self.cdltensor.device.device_id
        return (device_type, device_id)

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        """Produce a DLPack tensor from this array

        Parameters
        ----------
        stream : Optional[int]
            The stream to use for the DLPack tensor

        max_version : int, optional
            The maximum version of the DLPack tensor to produce

        dl_device : Optional[Tuple[int, int]]
            The device to use for the DLPack tensor

        copy : Optional[bool]
            Whether to copy the data to the new device

        Returns
        -------
        dlpack : DLPack tensor

        Raises
        ------
        BufferError
            Export failed
        """
        if max_version is None:
            # Keep and use the DLPack 0.X implementation
            # Note: from March 2025 onwards (but ideally as late as
            # possible), it's okay to raise BufferError here
            return self._to_dlpack()
        else:
            # We get to produce `DLManagedTensorVersioned` now. Note that
            # our_own_dlpack_version is the max version that the *producer*
            # supports and fills in the `DLManagedTensorVersioned::version`
            # field
            if max_version[0] >= __dlpack_version__[0]:
                if dl_device is not None and dl_device != self.__dlpack_device__():
                    raise BufferError("dl_device of different type not supported")
                if copy is not None and copy:
                    raise BufferError("copy not yet supported")
                return self._to_dlpack_versioned()
            elif max_version[0] < 1:
                return self.__ctypes_handle__to_dlpack()
            else:
                raise BufferError(f"Unsupported max_version {max_version}")


_set_class_tensor(Tensor)
_register_object_by_index(kTVMFFITensor, Tensor)


cdef int _dltensor_test_wrapper_c_dlpack_from_pyobject(
    void* obj, DLManagedTensorVersioned** out, TVMFFIStreamHandle* env_stream
) except -1:
    cdef PyObject* py_obj = <PyObject*>obj
    cdef DLTensorTestWrapper wrapper = <DLTensorTestWrapper>py_obj
    cdef TVMFFIStreamHandle current_stream
    cdef DLManagedTensorVersioned* temp_managed_tensor
    if env_stream != NULL:
        env_stream[0] = TVMFFIEnvGetStream(
            wrapper.tensor.cdltensor.device.device_type,
            wrapper.tensor.cdltensor.device.device_id
        )

    return TVMFFITensorToDLPackVersioned(wrapper.tensor.chandle, out)


def _dltensor_test_wrapper_c_dlpack_from_pyobject_as_intptr():
    cdef DLPackFromPyObject converter_func = _dltensor_test_wrapper_c_dlpack_from_pyobject
    cdef void* temp_ptr = <void*>converter_func
    cdef long long temp_int_ptr = <long long>temp_ptr
    return temp_int_ptr


cdef class DLTensorTestWrapper:
    """Wrapper of a Tensor that exposes DLPack protocol, only for testing purpose.
    """
    __c_dlpack_from_pyobject__ = _dltensor_test_wrapper_c_dlpack_from_pyobject_as_intptr()

    cdef Tensor tensor
    cdef dict __dict__
    def __init__(self, tensor):
        self.tensor = tensor

    def __tvm_ffi_env_stream__(self):
        cdef TVMFFIStreamHandle stream
        cdef long long stream_as_int
        cdef int c_api_ret_code
        stream = TVMFFIEnvGetStream(
            self.tensor.cdltensor.device.device_type, self.tensor.cdltensor.device.device_id)
        stream_as_int = <long long>stream
        return stream_as_int

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()

    def __dlpack__(self, *, **kwargs):
        return self.tensor.__dlpack__(**kwargs)


cdef inline object make_ret_dltensor(TVMFFIAny result):
    cdef DLTensor* dltensor
    dltensor = <DLTensor*>result.v_ptr
    tensor = _CLASS_TENSOR.__new__(_CLASS_TENSOR)
    (<Object>tensor).chandle = NULL
    (<Tensor>tensor).cdltensor = dltensor
    return tensor


cdef inline object make_tensor_from_chandle(TVMFFIObjectHandle chandle, DLPackToPyObject c_dlpack_to_pyobject = NULL):
    # TODO: Implement
    cdef Tensor tensor
    cdef void* py_obj
    cdef DLManagedTensorVersioned* dlpack

    if c_dlpack_to_pyobject != NULL:
        # try convert and import into the environment array if possible
        if TVMFFITensorToDLPackVersioned(chandle, &dlpack) == 0:
            try:
                # note that py_obj already holds an extra reference to the tensor
                # so we need to decref it after the conversion
                c_dlpack_to_pyobject(dlpack, &py_obj)
                tensor = <Tensor>(<PyObject*>py_obj)
                Py_DECREF(tensor)
                return tensor
            except Exception:
                pass
    # default return the tensor
    tensor = _CLASS_TENSOR.__new__(_CLASS_TENSOR)
    (<Object>tensor).chandle = chandle
    (<Tensor>tensor).cdltensor = TVMFFITensorGetDLTensorPtr(chandle)
    return tensor


cdef inline object make_tensor_from_any(TVMFFIAny any, DLPackToPyObject c_dlpack_to_pyobject):
    return make_tensor_from_chandle(any.v_ptr, c_dlpack_to_pyobject)
