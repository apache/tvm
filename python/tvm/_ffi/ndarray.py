# pylint: disable=invalid-name, unused-import
"""Runtime NDArray api"""
from __future__ import absolute_import

import sys
import ctypes
import numpy as np
from .base import _LIB, check_call, c_array, string_types, _FFI_MODE
from .runtime_ctypes import TVMType, TVMContext, TVMArray, TVMArrayHandle, tvm_shape_index_t


IMPORT_EXCEPT = RuntimeError if _FFI_MODE == "cython" else ImportError

try:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "ctypes":
        raise ImportError()
    if sys.version_info >= (3, 0):
        from ._cy3.core import _set_class_ndarray, _reg_dltensor, _make_array
        from ._cy3.core import NDArrayBase as _NDArrayBase
    else:
        from ._cy2.core import _set_class_ndarray, _reg_dltensor, _make_array
        from ._cy2.core import NDArrayBase as _NDArrayBase
except IMPORT_EXCEPT:
    # pylint: disable=wrong-import-position
    from ._ctypes.ndarray import _set_class_ndarray, _reg_dltensor, _make_array
    from ._ctypes.ndarray import NDArrayBase as _NDArrayBase


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
    ctx: TVMContext
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
        if not dev_type in TVMContext.STR2MASK:
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
    arr.dtype = TVMType(np.dtype(data.dtype).name)
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
    dtype = TVMType(dtype)
    check_call(_LIB.TVMArrayAlloc(
        shape, ndim, dtype, ctx, ctypes.byref(handle)))
    return _make_array(handle, False)

class NDArrayBase(_NDArrayBase):
    """A simple Device/CPU Array object in runtime."""
    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.handle.contents.shape[i] for i in range(self.handle.contents.ndim))

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
            self._sync_copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def _sync_copyfrom(self, source_array):
        """Peform an synchronize copy from the array.

        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.
        """
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=self.dtype)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported' % str(type(source_array)))
        source_array = np.ascontiguousarray(source_array, dtype=self.dtype)
        if source_array.shape != self.shape:
            raise ValueError('array shape do not match the shape of NDArray')
        source_tvm_arr, shape = numpyasarray(source_array)
        check_call(_LIB.TVMArrayCopyFromTo(
            ctypes.byref(source_tvm_arr), self.handle, None))
        # de-allocate shape until now
        _ = shape

    def asnumpy(self):
        """Convert this array to numpy array

        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        np_arr = np.empty(self.shape, dtype=self.dtype)
        tvm_arr, shape = numpyasarray(np_arr)
        check_call(_LIB.TVMArrayCopyFromTo(
            self.handle, ctypes.byref(tvm_arr), None))
        _ = shape
        return np_arr

    def copyto(self, target):
        """Copy array to target

        Parameters
        ----------
        target : tvm.NDArray
            The target array to be copied, must have same shape as this array.
        """
        if isinstance(target, TVMContext):
            target = empty(self.shape, self.dtype, target)
        if isinstance(target, NDArrayBase):
            check_call(_LIB.TVMArrayCopyFromTo(
                self.handle, target.handle, None))
        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))
        return target


def register_dltensor(cls):
    """Register a DLTensor compatible class to TVM.

    After the class is registered, the class will be able
    to directly pass as Function argument generated by TVM.

    Parameters
    ----------
    cls : class
        The class object to be registered as DLTensor compatible.

    Note
    ----
    The registered class is requires a property _dltensor_addr,
    which returns an integer that represents the address of DLTensor.

    Returns
    -------
    cls : class
        The class being registered.

    Example
    -------
    The following code registers user defined class
    MyTensor to be DLTensor compatible.

    .. code-block:: python

       @tvm.register_dltensor
       class MyTensor(object):
           def __init__(self):
               self.handle = _LIB.NewDLTensor()

           @property
           def _dltensor_addr(self):
               return self.handle.value
    """
    _reg_dltensor(cls)
    return cls
