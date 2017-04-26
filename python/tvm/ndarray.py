"""TVM Runtime NDArray API.

tvm.ndarray provides a minimum runtime array API to test
the correctness of the program.
"""
# pylint: disable=invalid-name,unused-import
from __future__ import absolute_import as _abs
import numpy as _np

from ._ffi.ndarray import TVMContext, TVMType, NDArrayBase
from ._ffi.ndarray import cpu, gpu, opencl, vpi, empty, sync
from ._ffi.ndarray import _init_ndarray_module

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


_init_ndarray_module(NDArray)
