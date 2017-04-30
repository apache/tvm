# pylint: disable=invalid-name, protected-access, too-many-arguments,  global-statement
# pylint: disable=attribute-defined-outside-init, no-member, missing-docstring
"""Runtime NDArray api"""
from __future__ import absolute_import
import ctypes
import numpy as np
from .base import _LIB, check_call, c_array, string_types
from .. import _api_internal

tvm_shape_index_t = ctypes.c_int64

class TVMByteArray(ctypes.Structure):
    """Temp data structure for byte array."""
    _fields_ = [("data", ctypes.POINTER(ctypes.c_byte)),
                ("size", ctypes.c_size_t)]

class TVMType(ctypes.Structure):
    """TVM datatype structure"""
    _fields_ = [("type_code", ctypes.c_uint8),
                ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]
    CODE2STR = {
        0 : 'int',
        1 : 'uint',
        2 : 'float',
        4 : 'handle'
    }
    def __init__(self, type_str, lanes=1):
        super(TVMType, self).__init__()
        if isinstance(type_str, np.dtype):
            type_str = str(type_str)
        if type_str.startswith("int"):
            self.type_code = 0
            bits = int(type_str[3:])
        elif type_str.startswith("uint"):
            self.type_code = 1
            bits = int(type_str[4:])
        elif type_str.startswith("float"):
            self.type_code = 2
            bits = int(type_str[5:])
        elif type_str.startswith("handle"):
            self.type_code = 4
            bits = 64
        else:
            raise ValueError("Donot know how to handle type %s" % type_str)

        bits = 32 if bits == 0 else bits
        if (bits & (bits - 1)) != 0 or bits < 8:
            raise ValueError("Donot know how to handle type %s" % type_str)
        self.bits = bits
        self.lanes = lanes

    def __repr__(self):
        x = "%s%d" % (TVMType.CODE2STR[self.type_code], self.bits)
        if self.lanes != 1:
            x += "x%d" % self.lanes
        return x

    def __eq__(self, other):
        return (self.bits == other.bits and
                self.type_code == other.type_code and
                self.lanes == other.lanes)

    def __ne__(self, other):
        return not self.__eq__(other)


class TVMContext(ctypes.Structure):
    """TVM context strucure."""
    _fields_ = [("device_id", ctypes.c_int),
                ("device_type", ctypes.c_int)]
    MASK2STR = {
        1 : 'cpu',
        2 : 'gpu',
        4 : 'opencl',
        8 : 'metal',
        9 : 'vpi'
    }
    STR2MASK = {
        'cpu': 1,
        'gpu': 2,
        'cuda': 2,
        'cl': 4,
        'opencl': 4,
        'metal': 8,
        'vpi': 9
    }
    def __init__(self, device_type, device_id):
        super(TVMContext, self).__init__()
        self.device_id = device_id
        self.device_type = device_type

    @property
    def exist(self):
        """Whether this device exist."""
        return _api_internal._GetDeviceAttr(
            self.device_type, self.device_id, 0) != 0

    @property
    def max_threads_per_block(self):
        """Maximum number of threads on each block."""
        return _api_internal._GetDeviceAttr(
            self.device_type, self.device_id, 1)

    @property
    def warp_size(self):
        """Number of threads that executes in concurrent."""
        return _api_internal._GetDeviceAttr(
            self.device_type, self.device_id, 2)

    def sync(self):
        """Synchronize until jobs finished at the context."""
        check_call(_LIB.TVMSynchronize(self, None))

    def __eq__(self, other):
        return (isinstance(other, TVMContext) and
                self.device_id == other.device_id and
                self.device_type == other.device_type)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "%s(%d)" % (
            TVMContext.MASK2STR[self.device_type], self.device_id)


class TVMArray(ctypes.Structure):
    """TVMValue in C API"""
    _fields_ = [("data", ctypes.c_void_p),
                ("ctx", TVMContext),
                ("ndim", ctypes.c_int),
                ("dtype", TVMType),
                ("shape", ctypes.POINTER(tvm_shape_index_t)),
                ("strides", ctypes.POINTER(tvm_shape_index_t)),
                ("byte_offset", ctypes.c_size_t)]


TVMArrayHandle = ctypes.POINTER(TVMArray)

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
    return _CLASS_NDARRAY(handle)


class NDArrayBase(object):
    """A simple Device/CPU Array object in runtime."""
    __slots__ = ["handle", "is_view"]
    # pylint: disable=no-member
    def __init__(self, handle, is_view=False):
        """Initialize the function with handle

        Parameters
        ----------
        handle : TVMArrayHandle
            the handle to the underlying C++ TVMArray
        """
        self.handle = handle
        self.is_view = is_view

    def __del__(self):
        if not self.is_view:
            check_call(_LIB.TVMArrayFree(self.handle))

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

def _make_array(handle, is_view):
    handle = ctypes.cast(handle, TVMArrayHandle)
    return _CLASS_NDARRAY(handle, is_view)

_CLASS_NDARRAY = None

def _set_class_ndarray(cls):
    global _CLASS_NDARRAY
    _CLASS_NDARRAY = cls
