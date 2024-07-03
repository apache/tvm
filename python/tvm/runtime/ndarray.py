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
# pylint: disable=invalid-name, unused-import, redefined-outer-name
"""Runtime NDArray API"""
import ctypes
import warnings
from typing import Optional

import numpy as np

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None
import tvm._ffi
from tvm._ffi.base import _FFI_MODE, _LIB, c_array, check_call, string_types
from tvm._ffi.runtime_ctypes import (
    DataType,
    DataTypeCode,
    Device,
    TVMArray,
    TVMArrayHandle,
    tvm_shape_index_t,
)

from . import _ffi_api

try:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "ctypes":
        raise ImportError()
    from tvm._ffi._cy3.core import (
        NDArrayBase,
        _from_dlpack,
        _make_array,
        _set_class_ndarray,
    )
except (RuntimeError, ImportError) as error:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "cython":
        raise error
    from tvm._ffi._ctypes.ndarray import (
        NDArrayBase,
        _from_dlpack,
        _make_array,
        _set_class_ndarray,
    )


@tvm._ffi.register_object("runtime.NDArray")
class NDArray(NDArrayBase):
    """Lightweight NDArray class of TVM runtime.

    Strictly this is only an Array Container (a buffer object)
    No arthimetic operations are defined.
    All operations are performed by TVM functions.

    The goal is not to re-build yet another array library.
    Instead, this is a minimal data structure to demonstrate
    how can we use TVM in existing project which might have their own array containers.
    """

    @property
    def dtype(self):
        """Type of this array"""
        return str(self.handle.contents.dtype)

    @property
    def device(self):
        """Device of this array"""
        return self.handle.contents.device

    def __dlpack__(self, stream=None):  # pylint: disable=unused-argument
        """Export the array for consumption by from_dlpack() as a DLPack capsule.

        Parameters
        ----------
        stream : int, optional
            A Python integer representing a pointer to a stream.
            Stream is provided by the consumer to the producer to instruct the producer
            to ensure that operations can safely be performed on the array.

        Returns
        -------
        capsule : PyCapsule
            A DLPack capsule for the array, containing a DLPackManagedTensor.
        """
        return self.to_dlpack()

    def __dlpack_device__(self):
        """Return a tuple of device_type, device_id in DLPack convention"""
        return (self.handle.contents.device.device_type, self.handle.contents.device.device_id)

    def __hash__(self):
        return ctypes.cast(self.handle, ctypes.c_void_p).value

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Check object identity equality

        Parameters
        ----------
        other : object
            The other object to compare to

        Returns
        -------
        same : bool
            Whether other is same as self.
        """
        if not isinstance(other, NDArrayBase):
            return False
        return self.__hash__() == other.__hash__()

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if (
            not isinstance(in_slice, slice)
            or in_slice.start is not None
            or in_slice.stop is not None
        ):
            raise ValueError("Array only support set from numpy array")
        if isinstance(value, NDArrayBase):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self.copyfrom(value)
        else:
            raise TypeError(f"type {type(value)} not supported")

    def copyfrom(self, source_array):
        """Perform a synchronous copy from the array.

        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.

        Returns
        -------
        arr : NDArray
            Reference to self.
        """
        if isinstance(source_array, NDArrayBase):
            source_array.copyto(self)
            return self

        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=self.dtype)
            except:
                raise TypeError(
                    f"array must be an array_like data, type {type(source_array)} is not supported"
                )

        t = DataType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)

        if source_array.shape != shape:
            raise ValueError(
                f"array shape do not match the shape of NDArray {source_array.shape} vs {shape}"
            )
        numpy_str_map = DataType.NUMPY2STR
        np_dtype_str = (
            numpy_str_map[source_array.dtype]
            if source_array.dtype in numpy_str_map
            else str(source_array.dtype)
        )
        if (not source_array.flags["C_CONTIGUOUS"]) or (
            dtype == "bfloat16" or dtype != np_dtype_str
        ):
            if dtype == "bfloat16":
                source_array = np.frombuffer(source_array.tobytes(), "uint16")
            source_array = np.ascontiguousarray(
                source_array, dtype="uint16" if dtype == "bfloat16" else dtype
            )
        assert source_array.flags["C_CONTIGUOUS"]
        data = source_array.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(source_array.size * source_array.dtype.itemsize)
        check_call(_LIB.TVMArrayCopyFromBytes(self.handle, data, nbytes))
        return self

    def __repr__(self):
        res = f"<tvm.nd.NDArray shape={self.shape}, {self.device}>\n"
        res += self.numpy().__repr__()
        return res

    def __str__(self):
        return str(self.numpy())

    def asnumpy(self):
        """Convert this array to numpy array. This API will be deprecated in TVM v0.8 release.
        Please use `numpy` instead."""
        warnings.warn(
            "NDArray.asnumpy() will be deprecated in TVM v0.8 release. "
            "Please use NDArray.numpy() instead.",
            DeprecationWarning,
        )
        return self.numpy()

    def numpy(self):
        """Convert this array to numpy array

        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        t = DataType(self.dtype)
        shape, dtype = self.shape, self.dtype
        old_dtype = dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)
        if dtype == "int4":
            dtype = "int8"
        if dtype == "bfloat16":
            dtype = "uint16"
        if dtype == "e4m3_float8":
            if ml_dtypes is not None:
                dtype = ml_dtypes.float8_e4m3fn
            else:
                raise RuntimeError(
                    "ml_dtypes is not installed, cannot convert e4m3_float8 array to numpy."
                )
        if dtype == "e5m2_float8":
            if ml_dtypes is not None:
                dtype = ml_dtypes.float8_e5m2
            else:
                raise RuntimeError(
                    "ml_dtypes is not installed, cannot convert e5m2_float8 array to numpy."
                )
        np_arr = np.empty(shape, dtype=dtype)
        assert np_arr.flags["C_CONTIGUOUS"]
        data = np_arr.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(np_arr.size * np_arr.dtype.itemsize)
        check_call(_LIB.TVMArrayCopyToBytes(self.handle, data, nbytes))
        if old_dtype == "int4":
            length = np_arr.size
            np_arr_ret = np.empty((length,), dtype="int8")
            np_arr = np_arr.reshape((length,))
            old_index = np.bitwise_and(np_arr, 0x0F)
            even_index = np.bitwise_and(np_arr >> 4, 0x0F)
            np_arr_ret[1::2] = old_index[0 : length // 2]
            np_arr_ret[0::2] = even_index[0 : length // 2]
            return np_arr_ret.reshape(shape)
        return np_arr

    def copyto(self, target, mem_scope=None):
        """Copy array to target

        Parameters
        ----------
        target : NDArray
            The target array to be copied, must have same shape as this array.

        mem_scope : Optional[str]
            The memory scope of the array.
        """
        if isinstance(target, NDArrayBase):
            return self._copyto(target)
        if isinstance(target, Device):
            res = empty(self.shape, self.dtype, target, mem_scope)
            return self._copyto(res)
        raise ValueError(f"Unsupported target type {type(target)}")

    def _create_view(self, shape, dtype: Optional[str] = None, relative_byte_offset: int = 0):
        """Create a view into an existing array.

        The view shares the same allocation and datatype as the
        existing array, but can have a different array shape.  This is
        useful for runtimes that support non-flat memory, where both
        the physical shape of an allocation and the logical shape of
        the tensor it represents may need to be independently
        specified.

        Warning: This function should not be used outside of low-level
        manipulations, as it breaks non-aliasing assumptions made by
        TVM.  This function may also be removed/replaced in the
        future.

        Parameters
        ----------
        shape: Union[tvm.runtime.ShapeTuple, Sequence[typing.SupportsInt]]

            The shape of the view.

        dtype: Optional[str]

            The datatype of the view.  If None (default), the view
            will be the same data type as the current array.

        relative_byte_offset: int

            The location of the view, relative to the location of the current
            array.

            Note: While the `DLTensor.byte_offset` field of the returned view
            is usually the same as `relative_byte_offset`, this is not
            guaranteed.  The `DLTensor.byte_offset` field is relative to the
            start of the backing allocation, while the `relative_byte_offset`
            is relative to the start of `self`.

        """

        if not isinstance(shape, tvm.runtime.ShapeTuple):
            shape = tvm.runtime.ShapeTuple([int(dim) for dim in shape])

        if dtype is None:
            dtype = self.dtype

        return _ffi_api.TVMArrayCreateView(self, shape, dtype, relative_byte_offset)


def device(dev_type, dev_id=0):
    """Construct a TVM device with given device type and id.

    Parameters
    ----------
    dev_type: int or str
        The device type mask or name of the device.

    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev: tvm.runtime.Device
        The corresponding device.

    Examples
    --------
    Device can be used to create reflection of device by
    string representation of the device type.

    .. code-block:: python

      assert tvm.device("cpu", 1) == tvm.cpu(1)
      assert tvm.device("cuda", 0) == tvm.cuda(0)
    """
    if isinstance(dev_type, Device):
        return dev_type
    if not isinstance(dev_id, int):
        raise ValueError(f"Invalid device id: {dev_id}")

    if isinstance(dev_type, string_types):
        dev_type = dev_type.split()[0]
        if dev_type.count(":") == 0:
            pass
        elif dev_type.count(":") == 1:
            # It will override the dev_id passed by the user.
            dev_type, dev_id = dev_type.split(":")
            if not dev_id.isdigit():
                raise ValueError(f"Invalid device id: {dev_id}")
            dev_id = int(dev_id)
        else:
            raise ValueError(f"Invalid device string: {dev_type}")

        if dev_type not in Device.STR2MASK:
            raise ValueError(f"Unknown device type: {dev_type}")

        return Device(Device.STR2MASK[dev_type], dev_id)
    return Device(dev_type, dev_id)


def numpyasarray(np_data):
    """Return a TVMArray representation of a numpy array."""
    data = np_data
    assert data.flags["C_CONTIGUOUS"]
    arr = TVMArray()
    shape = c_array(tvm_shape_index_t, data.shape)
    arr.data = data.ctypes.data_as(ctypes.c_void_p)
    arr.shape = shape
    arr.strides = None
    arr.dtype = DataType(np.dtype(data.dtype).name)
    arr.ndim = data.ndim
    # CPU device
    arr.device = device(Device.kDLCPU, 0)
    return arr, shape


def empty(shape, dtype="float32", device=device(Device.kDLCPU, 0), mem_scope=None):
    """Create an empty array given shape and device

    Parameters
    ----------
    shape : Union[tvm.runtime.ShapeTuple, Sequence[typing.SupportsInt]]
        The shape of the array.

    dtype : type or str
        The data type of the array.

    device : Device
        The device of the array.

    mem_scope : Optional[str]
        The memory scope of the array.

    Returns
    -------
    arr : tvm.nd.NDArray
        The array tvm supported.
    """
    if not isinstance(shape, tvm.runtime.ShapeTuple):
        shape = tvm.runtime.ShapeTuple([int(dim) for dim in shape])
    dtype = DataType(dtype)
    arr = _ffi_api.TVMArrayAllocWithScope(shape, dtype, device, mem_scope)
    return arr


def from_dlpack(dltensor):
    """Produces an array from an object with __dlpack__ method or a DLPack tensor w/o memory copy.
    Retreives the underlying DLPack tensor's pointer to create an array from the
    data. Removes the original DLPack tensor's destructor as now the array is
    responsible for destruction.

    Parameters
    ----------
    dltensor : object with __dlpack__ attribute or a DLPack capsule

    Returns
    -------
    arr: tvm.nd.NDArray
        The array view of the tensor data.
    """
    t = type(dltensor)
    if t.__module__ == "builtins" and t.__name__ == "PyCapsule":
        return _from_dlpack(dltensor)

    if hasattr(dltensor, "__dlpack__"):
        dlpack_caps = dltensor.__dlpack__()
        return _from_dlpack(dlpack_caps)
    raise AttributeError("Required attribute __dlpack__ not found")


def cpu(dev_id=0):
    """Construct a CPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(Device.kDLCPU, dev_id)


def cuda(dev_id=0):
    """Construct a CUDA GPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(Device.kDLCUDA, dev_id)


def gpu(dev_id=0):
    """Construct a CUDA GPU device

        deprecated:: 0.9.0
        Use :py:func:`tvm.cuda` instead.

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    warnings.warn(
        "Please use tvm.cuda() instead of tvm.gpu(). tvm.gpu() is going to be deprecated in 0.9.0"
    )
    return Device(Device.kDLCUDA, dev_id)


def rocm(dev_id=0):
    """Construct a ROCM device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(Device.kDLROCM, dev_id)


def opencl(dev_id=0):
    """Construct a OpenCL device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(Device.kDLOpenCL, dev_id)


def metal(dev_id=0):
    """Construct a metal device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(Device.kDLMetal, dev_id)


def vpi(dev_id=0):
    """Construct a VPI simulated device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(Device.kDLVPI, dev_id)


def vulkan(dev_id=0):
    """Construct a Vulkan device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(Device.kDLVulkan, dev_id)


def ext_dev(dev_id=0):
    """Construct a extension device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device

    Note
    ----
    This API is reserved for quick testing of new
    device by plugin device API as ext_dev.
    """
    return Device(Device.kDLExtDev, dev_id)


def hexagon(dev_id=0):
    """Construct a Hexagon device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(Device.kDLHexagon, dev_id)


def webgpu(dev_id=0):
    """Construct a webgpu device.

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(Device.kDLWebGPU, dev_id)


cl = opencl
mtl = metal


def array(arr, device=cpu(0), mem_scope=None):
    """Create an array from source arr.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to be copied from

    device : Device, optional
        The device to create the array

    mem_scope : Optional[str]
        The memory scope of the array

    Returns
    -------
    ret : NDArray
        The created array
    """
    if isinstance(arr, tvm.ir.container.Array):
        raise AttributeError("arr is an instance of", type(arr))

    if not isinstance(arr, (np.ndarray, NDArray)):
        arr = np.array(arr)
    return empty(arr.shape, arr.dtype, device, mem_scope).copyfrom(arr)


# Register back to FFI
_set_class_ndarray(NDArray)
