# coding: utf-8
# pylint: disable=invalid-name, unused-import
""" ctypes library of nnvm and helper functions """
from __future__ import absolute_import

import os
import sys
import ctypes
import numpy as np
from . import libinfo

try:
    import tvm
except ImportError:
    pass

#----------------------------
# library loading
#----------------------------
if sys.version_info[0] == 3:
    string_types = str
    numeric_types = (float, int, np.float32, np.int32)
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    py_str = lambda x: x.decode('utf-8')
else:
    string_types = basestring
    numeric_types = (float, int, long, np.float32, np.int32)
    py_str = lambda x: x


class NNVMError(Exception):
    """Error that will be throwed by all nnvm functions"""
    pass

def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path()
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_LOCAL)
    # DMatrix functions
    lib.NNGetLastError.restype = ctypes.c_char_p
    return lib

# version number
__version__ = libinfo.__version__
# library instance of nnvm
_LIB = _load_lib()
# The FFI mode of TVM
_FFI_MODE = os.environ.get("TVM_FFI", "auto")

# type definitions
nn_uint = ctypes.c_uint
OpHandle = ctypes.c_void_p
SymbolHandle = ctypes.c_void_p
GraphHandle = ctypes.c_void_p

#----------------------------
# helper function definition
#----------------------------
def check_call(ret):
    """Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret != 0:
        raise NNVMError(py_str(_LIB.NNGetLastError()))

def c_str(string):
    """Create ctypes char * from a python string
    Parameters
    ----------
    string : string type
        python string

    Returns
    -------
    str : c_char_p
        A char pointer that can be passed to C API
    """
    return ctypes.c_char_p(string.encode('utf-8'))


def c_array(ctype, values):
    """Create ctypes array from a python array

    Parameters
    ----------
    ctype : ctypes data type
        data type of the array we want to convert to

    values : tuple or list
        data content

    Returns
    -------
    out : ctypes array
        Created ctypes array
    """
    return (ctype * len(values))(*values)

def ctypes2buffer(cptr, length):
    """Convert ctypes pointer to buffer type.

    Parameters
    ----------
    cptr : ctypes.POINTER(ctypes.c_char)
        pointer to the raw memory region
    length : int
        the length of the buffer

    Returns
    -------
    buffer : bytearray
        The raw byte memory buffer
    """
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_char)):
        raise TypeError('expected char pointer')
    res = bytearray(length)
    rptr = (ctypes.c_char * length).from_buffer(res)
    if not ctypes.memmove(rptr, cptr, length):
        raise RuntimeError('memmove failed')
    return res

def ctypes2numpy_shared(cptr, shape):
    """Convert a ctypes pointer to a numpy array

    The result numpy array shares the memory with the pointer

    Parameters
    ----------
    cptr : ctypes.POINTER(mx_float)
        pointer to the memory region

    shape : tuple
        shape of target ndarray

    Returns
    -------
    out : numpy_array
        A numpy array : numpy array
    """
    if not isinstance(cptr, ctypes.POINTER(mx_float)):
        raise RuntimeError('expected float pointer')
    size = 1
    for s in shape:
        size *= s
    dbuffer = (mx_float * size).from_address(ctypes.addressof(cptr.contents))
    return np.frombuffer(dbuffer, dtype=np.float32).reshape(shape)


def ctypes2docstring(num_args, arg_names, arg_types, arg_descs, remove_dup=True):
    """Convert ctypes returned doc string information into parameters docstring.

    num_args : nn_uint
        Number of arguments.

    arg_names : ctypes.POINTER(ctypes.c_char_p)
        Argument names.

    arg_types : ctypes.POINTER(ctypes.c_char_p)
        Argument type information.

    arg_descs : ctypes.POINTER(ctypes.c_char_p)
        Argument description information.

    remove_dup : boolean, optional
        Whether remove duplication or not.

    Returns
    -------
    docstr : str
        Python docstring of parameter sections.
    """
    param_keys = set()
    param_str = []
    for i in range(num_args.value):
        key = py_str(arg_names[i])
        if key in param_keys and remove_dup:
            continue
        param_keys.add(key)
        type_info = py_str(arg_types[i])
        ret = '%s : %s' % (key, type_info)
        if arg_descs[i]:
            ret += '\n    ' + py_str(arg_descs[i])
        param_str.append(ret)
    doc_str = ('Parameters\n' +
               '----------\n' +
               '%s\n')
    doc_str = doc_str % ('\n'.join(param_str))
    return doc_str
