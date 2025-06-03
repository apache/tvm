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

from tvm.runtime import Device

import tvm.ffi
from . import _ffi_api


from ..ffi import (
    device,
    cpu,
    cuda,
    rocm,
    opencl,
    metal,
    vpi,
    vulkan,
    ext_dev,
    hexagon,
    webgpu,
)


def from_dlpack(ext_tensor):
    """
    Convert an external tensor to an NDArray.

    Parameters
    ----------
    ext_tensor : object
        The external tensor to convert.

    required_alignment : int
        The minimum required alignment to check for the tensor.

    required_contiguous : bool
        Whether to check for contiguous memory.
    """
    return tvm.ffi.from_dlpack(
        ext_tensor,
        required_alignment=64,
        required_contiguous=True,
    )


@tvm.ffi.register_object("object.NDArray")
class NDArray(tvm.ffi.core.NDArray):
    """Lightweight NDArray class of TVM runtime.

    Strictly this is only an Array Container (a buffer object)
    No arthimetic operations are defined.
    All operations are performed by TVM functions.

    The goal is not to re-build yet another array library.
    Instead, this is a minimal data structure to demonstrate
    how can we use TVM in existing project which might have their own array containers.
    """

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if (
            not isinstance(in_slice, slice)
            or in_slice.start is not None
            or in_slice.stop is not None
        ):
            raise ValueError("Array only support set from numpy array")
        if isinstance(value, NDArray):
            if not value.same_as(self):
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
        if isinstance(source_array, NDArray):
            source_array.copyto(self)
            return self

        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=self.dtype)
            except:
                raise TypeError(
                    f"array must be an array_like data, type {type(source_array)} is not supported"
                )

        t = tvm.ffi.dtype(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t = t.with_lanes(1)
            dtype = str(t)

        if source_array.shape != shape:
            raise ValueError(
                f"array shape do not match the shape of NDArray {source_array.shape} vs {shape}"
            )
        numpy_str_map = tvm.ffi.dtype.NUMPY_DTYPE_TO_STR
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
        if self.dtype.startswith("float4_e2m1fn"):
            # we need to pack the input data when converting to float4_e2m1fn type,
            data_bits = source_array.view(dtype="uint8").flatten()
            if data_bits.size % 2:
                data_bits = np.pad(data_bits, (0, 1), mode="constant", constant_values=0)
            data_bits = data_bits.reshape(-1, 2)
            packed = ((data_bits[:, 0] & 0x0F) << 4) | (data_bits[:, 1] & 0x0F)
            source_array = packed.astype(np.int8)
        assert source_array.flags["C_CONTIGUOUS"]
        data = source_array.ctypes.data_as(ctypes.c_void_p)
        nbytes = source_array.size * source_array.dtype.itemsize
        _ffi_api.TVMArrayCopyFromBytes(self, data, nbytes)
        return self

    def __repr__(self):
        res = f"<tvm.nd.NDArray shape={self.shape}, {self.device}>\n"
        res += self.numpy().__repr__()
        return res

    def __str__(self):
        return str(self.numpy())

    def numpy(self):
        """Convert this array to numpy array

        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        t = tvm.ffi.dtype(self.dtype)
        shape, dtype = self.shape, self.dtype
        old_dtype = dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t = t.with_lanes(1)
            dtype = str(t)
        if dtype == "int4":
            dtype = "int8"
        if dtype in [
            "bfloat16",
            "float8_e3m4",
            "float8_e4m3",
            "float8_e4m3b11fnuz",
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
            "float8_e8m0fnu",
            "float6_e2m3fn",
            "float6_e3m2fn",
            "float4_e2m1fn",
        ]:
            if ml_dtypes is None:
                raise RuntimeError(
                    f"ml_dtypes is not installed, cannot convert {dtype} array to numpy."
                )
            try:
                dtype = getattr(ml_dtypes, dtype)
            except AttributeError:
                raise RuntimeError(f"ml_dtypes has no attribute '{dtype}', cannot convert array.")
        np_arr = np.empty(shape, dtype=dtype)
        assert np_arr.flags["C_CONTIGUOUS"]
        data = np_arr.ctypes.data_as(ctypes.c_void_p)
        # TODO(kathy): revisit and get a mirrored function of ffi::GetDataSize
        # in Python to replace line below
        nbytes = np_arr.size if dtype == "bool" else (np_arr.size * old_dtype.bits + 7) // 8
        _ffi_api.TVMArrayCopyToBytes(self, data, nbytes)

        if old_dtype == "int4" or old_dtype.startswith("float4_e2m1fn"):
            length = np_arr.size
            np_arr = np_arr.view("int8")
            np_arr_ret = np.empty((length,), dtype="int8")
            np_arr = np_arr.reshape((length,))
            odd_index = np.bitwise_and(np_arr, 0x0F)
            even_index = np.bitwise_and(np_arr >> 4, 0x0F)
            np_arr_ret[1::2] = odd_index[0 : length // 2]
            np_arr_ret[0::2] = even_index[0 : (length + 1) // 2]
            return np_arr_ret.reshape(shape).view(dtype)

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
        if isinstance(target, NDArray):
            return self._copyto(target)
        if isinstance(target, tvm.ffi.core.Device):
            res = empty(self.shape, self.dtype, target, mem_scope)
            return self._copyto(res)
        raise ValueError(f"Unsupported target type {type(target)}")

    def _copyto(self, target_nd):
        """Internal function that implements copy to target ndarray."""
        _ffi_api.TVMArrayCopyFromTo(self, target_nd)
        return target_nd

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


def empty(shape, dtype="float32", device=None, mem_scope=None):
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
    device = device or cpu()
    if not isinstance(shape, tvm.runtime.ShapeTuple):
        shape = tvm.runtime.ShapeTuple([int(dim) for dim in shape])
    dtype = tvm.ffi.dtype(dtype)
    arr = _ffi_api.TVMArrayAllocWithScope(shape, dtype, device, mem_scope)
    return arr


def array(arr, device=None, mem_scope=None):
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
    device = device or cpu()

    if not isinstance(arr, (np.ndarray, NDArray)):
        arr = np.array(arr)
    return empty(arr.shape, arr.dtype, device, mem_scope).copyfrom(arr)


# Register back to FFI
tvm.ffi.core._set_class_ndarray(NDArray)
