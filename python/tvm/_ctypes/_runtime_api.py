# pylint: disable=invalid-name, protected-access, too-many-arguments,  global-statement
# pylint: disable=attribute-defined-outside-init, no-member, missing-docstring
"""Symbolic configuration API."""
from __future__ import absolute_import as _abs

import ctypes
import numpy as np

from .._base import _LIB
from .._base import c_array, c_str
from .._base import check_call


tvm_index_t = ctypes.c_uint32


class TVMContext(ctypes.Structure):
    """TVM context strucure."""
    _fields_ = [("dev_mask", ctypes.c_int),
                ("dev_id", ctypes.c_int)]
    MASK2STR = {
        1 : 'cpu',
        2 : 'gpu',
        4 : 'opencl'
    }
    def __init__(self, dev_mask, dev_id):
        super(TVMContext, self).__init__()
        self.dev_mask = dev_mask
        self.dev_id = dev_id

    def __repr__(self):
        return "%s(%d)" % (
            TVMContext.MASK2STR[self.dev_mask], self.dev_id)

    @property
    def enabled(self):
        ret = ctypes.c_int()
        check_call(_LIB.TVMContextEnabled(self, ctypes.byref(ret)))
        return ret.value != 0


def cpu(dev_id=0):
    """Construct a CPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id
    """
    return TVMContext(1, dev_id)


def gpu(dev_id=0):
    """Construct a CPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id
    """
    return TVMContext(2, dev_id)


def opencl(dev_id=0):
    """Construct a OpenCL device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id
    """
    return TVMContext(4, dev_id)


class TVMDataType(ctypes.Structure):
    """TVM datatype structure"""
    _fields_ = [("type_code", ctypes.c_uint8),
                ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]
    CODE2STR = {
        0 : 'int',
        1 : 'uint',
        2 : 'float'
    }
    def __init__(self, type_str, lanes=1):
        super(TVMDataType, self).__init__()
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
        else:
            raise ValueError("Donot know how to handle type %s" % type_str)
        bits = 32 if bits == 0 else bits
        if (bits & (bits - 1)) != 0 or bits < 8:
            raise ValueError("Donot know how to handle type %s" % type_str)
        self.bits = bits
        self.lanes = lanes

    def __repr__(self):
        x = "%s%d" % (TVMDataType.CODE2STR[self.type_code], self.bits)
        if self.lanes != 1:
            x += "x%d" % self.lanes
        return x


class TVMArray(ctypes.Structure):
    """TVMArg in C API"""
    _fields_ = [("data", ctypes.c_void_p),
                ("shape", ctypes.POINTER(tvm_index_t)),
                ("strides", ctypes.POINTER(tvm_index_t)),
                ("ndim", tvm_index_t),
                ("dtype", TVMDataType),
                ("ctx", TVMContext)]

TVMArrayHandle = ctypes.POINTER(TVMArray)


def numpyasarray(np_data):
    """Return a TVMArray representation of a numpy array.
    """
    data = np_data
    assert data.flags['C_CONTIGUOUS']
    arr = TVMArray()
    shape = c_array(tvm_index_t, data.shape)
    arr.data = data.ctypes.data_as(ctypes.c_void_p)
    arr.shape = shape
    arr.strides = None
    arr.dtype = TVMDataType(np.dtype(data.dtype).name)
    arr.ndim = data.ndim
    # CPU device
    arr.ctx = cpu(0)
    return arr, shape


_ndarray_cls = None


def empty(shape, dtype="float32", ctx=cpu(0)):
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
    shape = c_array(tvm_index_t, shape)
    ndim = tvm_index_t(len(shape))
    handle = TVMArrayHandle()
    dtype = TVMDataType(dtype)
    check_call(_LIB.TVMArrayAlloc(
        shape, ndim, dtype, ctx, ctypes.byref(handle)))
    return _ndarray_cls(handle)


def sync(ctx):
    """Synchronize all the context

    Parameters
    ----------
    ctx : TVMContext
        The context to be synced
    """
    check_call(_LIB.TVMSynchronize(ctx, None))


def init_opencl(**kwargs):
    """Initialize the opencl with the options.

    Parameters
    ----------
    kwargs : dict
        The options
    """
    keys = []
    vals = []
    for k, v in kwargs.items():
        keys.append(c_str(k))
        vals.append(c_str(v))
    dev_mask = ctypes.c_int(4)
    out_code = ctypes.c_int()
    check_call(_LIB.TVMDeviceInit(
        dev_mask,
        c_array(ctypes.c_char_p, keys),
        c_array(ctypes.c_char_p, vals),
        ctypes.c_int(len(keys)),
        ctypes.byref(out_code)))
    return out_code.value != 0


class NDArrayBase(object):
    """A simple Device/CPU Array object in runtime."""
    __slots__ = ["handle"]
    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : TVMArrayHandle
            the handle to the underlying C++ TVMArray
        """
        self.handle = handle

    def __del__(self):
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


def _init_runtime_module(ndarray_class):
    global _ndarray_cls
    _ndarray_cls = ndarray_class
