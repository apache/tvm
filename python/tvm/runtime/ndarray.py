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
# pylint: disable=invalid-name, unused-import
"""Runtime NDArray API"""
import ctypes
import numpy as np
import tvm._ffi

from tvm._ffi.base import _LIB, check_call, c_array, string_types, _FFI_MODE
from tvm._ffi.runtime_ctypes import DataType, TVMContext, TVMArray, TVMArrayHandle
from tvm._ffi.runtime_ctypes import TypeCode, tvm_shape_index_t

try:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "ctypes":
        raise ImportError()
    from tvm._ffi._cy3.core import _set_class_ndarray, _make_array, _from_dlpack
    from tvm._ffi._cy3.core import NDArrayBase
except (RuntimeError, ImportError):
    # pylint: disable=wrong-import-position
    from tvm._ffi._ctypes.ndarray import _set_class_ndarray, _make_array, _from_dlpack
    from tvm._ffi._ctypes.ndarray import NDArrayBase


@tvm._ffi.register_object
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
    def ctx(self):
        """context of this array"""
        return self.handle.contents.ctx

    @property
    def context(self):
        """context of this array"""
        return self.ctx

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
        if (not isinstance(in_slice, slice) or
                in_slice.start is not None
                or in_slice.stop is not None):
            raise ValueError('Array only support set from numpy array')
        if isinstance(value, NDArrayBase):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self.copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

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
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported' % str(type(source_array)))

        t = DataType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)

        if source_array.shape != shape:
            raise ValueError("array shape do not match the shape of NDArray {0} vs {1}".format(
                source_array.shape, shape))
        source_array = np.ascontiguousarray(source_array, dtype=dtype)
        assert source_array.flags['C_CONTIGUOUS']
        data = source_array.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(source_array.size * source_array.dtype.itemsize)
        check_call(_LIB.TVMArrayCopyFromBytes(self.handle, data, nbytes))
        return self

    def __repr__(self):
        res = "<tvm.nd.NDArray shape={0}, {1}>\n".format(self.shape, self.context)
        res += self.asnumpy().__repr__()
        return res

    def __str__(self):
        return str(self.asnumpy())

    def asnumpy(self):
        """Convert this array to numpy array

        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        t = DataType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)
        np_arr = np.empty(shape, dtype=dtype)
        assert np_arr.flags['C_CONTIGUOUS']
        data = np_arr.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(np_arr.size * np_arr.dtype.itemsize)
        check_call(_LIB.TVMArrayCopyToBytes(self.handle, data, nbytes))
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
        if isinstance(target, TVMContext):
            res = empty(self.shape, self.dtype, target)
            return self._copyto(res)
        raise ValueError("Unsupported target type %s" % str(type(target)))


def context(dev_type, dev_id=0):
    """Construct a TVM context with given device type and id.

    Parameters
    ----------
    dev_type: int or str
        The device type mask or name of the device.

    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx: tvm.runtime.TVMContext
        The corresponding context.

    Examples
    --------
    Context can be used to create reflection of context by
    string representation of the device type.

    .. code-block:: python

      assert tvm.context("cpu", 1) == tvm.cpu(1)
      assert tvm.context("gpu", 0) == tvm.gpu(0)
      assert tvm.context("cuda", 0) == tvm.gpu(0)
    """
    if isinstance(dev_type, string_types):
        if '-device=micro_dev' in dev_type:
            dev_type = 'micro_dev'
        else:
            dev_type = dev_type.split()[0]
            if dev_type not in TVMContext.STR2MASK:
                raise ValueError("Unknown device type %s" % dev_type)
            dev_type = TVMContext.STR2MASK[dev_type]
    return TVMContext(dev_type, dev_id)


def numpyasarray(np_data):
    """Return a TVMArray representation of a numpy array.
    """
    data = np_data
    assert data.flags['C_CONTIGUOUS']
    arr = TVMArray()
    shape = c_array(tvm_shape_index_t, data.shape)
    arr.data = data.ctypes.data_as(ctypes.c_void_p)
    arr.shape = shape
    arr.strides = None
    arr.dtype = DataType(np.dtype(data.dtype).name)
    arr.ndim = data.ndim
    # CPU device
    arr.ctx = context(1, 0)
    return arr, shape


def empty(shape, dtype="float32", ctx=context(1, 0)):
    """Create an empty array given shape and device

    Parameters
    ----------
    shape : tuple of int
        The shape of the array

    dtype : type or str
        The data type of the array.

    ctx : TVMContext
        The context of the array

    Returns
    -------
    arr : tvm.nd.NDArray
        The array tvm supported.
    """
    shape = c_array(tvm_shape_index_t, shape)
    ndim = ctypes.c_int(len(shape))
    handle = TVMArrayHandle()
    dtype = DataType(dtype)
    check_call(_LIB.TVMArrayAlloc(
        shape, ndim,
        ctypes.c_int(dtype.type_code),
        ctypes.c_int(dtype.bits),
        ctypes.c_int(dtype.lanes),
        ctx.device_type,
        ctx.device_id,
        ctypes.byref(handle)))
    return _make_array(handle, False, False)


def from_dlpack(dltensor):
    """Produce an array from a DLPack tensor without memory copy.
    Retreives the underlying DLPack tensor's pointer to create an array from the
    data. Removes the original DLPack tensor's destructor as now the array is
    responsible for destruction.

    Parameters
    ----------
    dltensor : DLPack tensor
        Input DLManagedTensor, can only be consumed once.

    Returns
    -------
    arr: tvm.nd.NDArray
        The array view of the tensor data.
    """
    return _from_dlpack(dltensor)


def cpu(dev_id=0):
    """Construct a CPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context
    """
    return TVMContext(1, dev_id)


def gpu(dev_id=0):
    """Construct a GPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context
    """
    return TVMContext(2, dev_id)

def rocm(dev_id=0):
    """Construct a ROCM device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context
    """
    return TVMContext(10, dev_id)


def opencl(dev_id=0):
    """Construct a OpenCL device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context
    """
    return TVMContext(4, dev_id)


def metal(dev_id=0):
    """Construct a metal device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context
    """
    return TVMContext(8, dev_id)


def vpi(dev_id=0):
    """Construct a VPI simulated device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context
    """
    return TVMContext(9, dev_id)


def vulkan(dev_id=0):
    """Construct a Vulkan device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context
    """
    return TVMContext(7, dev_id)


def opengl(dev_id=0):
    """Construct a OpenGL device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context
    """
    return TVMContext(11, dev_id)


def ext_dev(dev_id=0):
    """Construct a extension device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context

    Note
    ----
    This API is reserved for quick testing of new
    device by plugin device API as ext_dev.
    """
    return TVMContext(12, dev_id)


def micro_dev(dev_id=0):
    """Construct a micro device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context
    """
    return TVMContext(13, dev_id)


def hexagon(dev_id=0):
    """Construct a Hexagon device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context
    """
    return TVMContext(14, dev_id)


cl = opencl
mtl = metal


def array(arr, ctx=cpu(0)):
    """Create an array from source arr.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to be copied from

    ctx : TVMContext, optional
        The device context to create the array

    Returns
    -------
    ret : NDArray
        The created array
    """
    if not isinstance(arr, (np.ndarray, NDArray)):
        arr = np.array(arr)
    return empty(arr.shape, arr.dtype, ctx).copyfrom(arr)

# Register back to FFI
_set_class_ndarray(NDArray)
