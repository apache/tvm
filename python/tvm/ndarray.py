"""TVM Runtime API.

This is a simplified runtime API for quick testing and proptyping.
"""
# pylint: disable=invalid-name,unused-import
from __future__ import absolute_import as _abs
import numpy as _np

from ._ctypes._ndarray import TVMContext, TVMType, NDArrayBase
from ._ctypes._ndarray import cpu, gpu, opencl, empty, sync
from ._ctypes._ndarray import _init_ndarray_module
from ._ctypes._function import Function

cl = opencl

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


def array(arr, ctx=cpu(0)):
    """Create an array from source arr.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to be copied from

    ctx : TVMContext
        The device context to create the array

    Returns
    -------
    ret : tvm.nd.NDArray
        The created array
    """
    if not isinstance(arr, _np.ndarray):
        arr = _np.array(arr)
    ret = empty(arr.shape, arr.dtype, ctx)
    ret[:] = arr
    return ret


_init_ndarray_module(NDArray)
