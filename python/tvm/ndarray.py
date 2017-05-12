"""TVM Runtime NDArray API.

tvm.ndarray provides a minimum runtime array API to test
the correctness of the program.
"""
# pylint: disable=invalid-name,unused-import
from __future__ import absolute_import as _abs
import numpy as _np

from ._ffi.ndarray import TVMContext, TVMType, NDArrayBase
from ._ffi.ndarray import context, empty
from ._ffi.ndarray import _set_class_ndarray, register_dltensor

class NDArray(NDArrayBase):
    """Lightweight NDArray class of TVM runtime.

    Strictly this is only an Array Container(a buffer object)
    No arthimetic operations are defined.
    All operations are performed by TVM functions.

    The goal is not to re-build yet another array library.
    Instead, this is a minimal data structure to demonstrate
    how can we use TVM in existing project which might have their own array containers.
    """
    pass


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


def metal(dev_id=0):
    """Construct a metal device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id
    """
    return TVMContext(8, dev_id)


def vpi(dev_id=0):
    """Construct a VPI simulated device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id
    """
    return TVMContext(9, dev_id)

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
    if not isinstance(arr, _np.ndarray):
        arr = _np.array(arr)
    ret = empty(arr.shape, arr.dtype, ctx)
    ret[:] = arr
    return ret

_set_class_ndarray(NDArray)
