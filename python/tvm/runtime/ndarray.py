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
import numpy as np
import tvm._ffi

from tvm._ffi.base import _LIB, check_call, c_array, string_types, _FFI_MODE
from tvm._ffi.runtime_ctypes import DataType, Device, TVMArray, TVMArrayHandle
from tvm._ffi.runtime_ctypes import DataTypeCode, tvm_shape_index_t
from . import _ffi_api

try:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "ctypes":
        raise ImportError()
    from tvm._ffi._cy3.core import _set_class_ndarray, _make_array, _from_dlpack
    from tvm._ffi._cy3.core import NDArrayBase
except (RuntimeError, ImportError) as error:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "cython":
        raise error
    from tvm._ffi._ctypes.ndarray import _set_class_ndarray, _make_array, _from_dlpack
    from tvm._ffi._ctypes.ndarray import NDArrayBase


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
            raise TypeError("type %s not supported" % str(type(value)))

    def copyfrom(self, source_array):
        """Peform an synchronize copy from the array.

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
                    "array must be an array_like data,"
                    + "type %s is not supported" % str(type(source_array))
                )

        t = DataType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)

        if source_array.shape != shape:
            raise ValueError(
                "array shape do not match the shape of NDArray {0} vs {1}".format(
                    source_array.shape, shape
                )
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
            source_array = np.ascontiguousarray(
                source_array, dtype="uint16" if dtype == "bfloat16" else dtype
            )
        assert source_array.flags["C_CONTIGUOUS"]
        data = source_array.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(source_array.size * source_array.dtype.itemsize)
        check_call(_LIB.TVMArrayCopyFromBytes(self.handle, data, nbytes))
        return self

    def __repr__(self):
        res = "<tvm.nd.NDArray shape={0}, {1}>\n".format(self.shape, self.device)
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

    def copyto(self, target):
        """Copy array to target

        Parameters
        ----------
        target : NDArray
            The target array to be copied, must have same shape as this array.
        """
        if isinstance(target, NDArrayBase):
            return self._copyto(target)
        if isinstance(target, Device):
            res = empty(self.shape, self.dtype, target)
            return self._copyto(res)
        raise ValueError("Unsupported target type %s" % str(type(target)))


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
    if isinstance(dev_type, string_types):
        dev_type = dev_type.split()[0]
        if dev_type not in Device.STR2MASK:
            raise ValueError("Unknown device type %s" % dev_type)
        dev_type = Device.STR2MASK[dev_type]
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
    arr.device = device(1, 0)
    return arr, shape


def empty(shape, dtype="float32", device=device(1, 0), mem_scope=None):
    """Create an empty array given shape and device

    Parameters
    ----------
    shape : tuple of int
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
    shape_imm = []
    for s in shape:
        if isinstance(s, tvm.tir.IntImm):
            shape_imm.append(s.value)
        else:
            shape_imm.append(int(s))
    arr = np.array(shape_imm, "int64")
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    shape_ptr = ctypes.cast(ptr, ctypes.c_void_p)
    ndim = len(shape_imm)
    dtype = DataType(dtype)
    arr = _ffi_api.TVMArrayAllocWithScope(shape_ptr, ndim, dtype, device, mem_scope)
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
    return Device(1, dev_id)


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
    return Device(2, dev_id)


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
        "Please use tvm.cuda() instead of tvm.gpu(). tvm.gpu() is going to be deprecated in 0.9.0",
    )
    return Device(2, dev_id)


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
    return Device(10, dev_id)


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
    return Device(4, dev_id)


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
    return Device(8, dev_id)


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
    return Device(9, dev_id)


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
    return Device(7, dev_id)


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
    return Device(12, dev_id)


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
    return Device(14, dev_id)


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
    return Device(15, dev_id)


cl = opencl
mtl = metal


def array(arr, device=cpu(0)):
    """Create an array from source arr.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to be copied from

    device : Device, optional
        The device device to create the array

    Returns
    -------
    ret : NDArray
        The created array
    """
    if isinstance(arr, tvm.ir.container.Array):
        raise AttributeError("arr is an instance of", type(arr))

    if not isinstance(arr, (np.ndarray, NDArray)):
        arr = np.array(arr)
    return empty(arr.shape, arr.dtype, device).copyfrom(arr)


# Register back to FFI
_set_class_ndarray(NDArray)
