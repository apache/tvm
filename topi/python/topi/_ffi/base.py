# coding: utf-8
# pylint: disable=invalid-name
""" ctypes library of TOPI and helper functions """
from __future__ import absolute_import

import sys
import os
import ctypes
import numpy as np
from tvm._ffi import libinfo

def get_lib_names():
    if sys.platform.startswith('win32'):
        return ['libtvm_topi.dll', 'tvm_topi.dll']
    if sys.platform.startswith('darwin'):
        return ['libtvm_topi.dylib', 'tvm_topi.dylib']
    return ['libtvm_topi.so', 'tvm_topi.so']

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
lib_search = os.path.join(curr_path, "..")

#----------------------------
# library loading
#----------------------------
if sys.version_info[0] == 3:
    string_types = (str,)
    numeric_types = (float, int, np.float32, np.int32)
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    py_str = lambda x: x.decode('utf-8')
else:
    string_types = (basestring,)
    numeric_types = (float, int, long, np.float32, np.int32)
    py_str = lambda x: x


class TVMError(Exception):
    """Error thrown by TVM function"""
    pass


def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path(get_lib_names(), lib_search)
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    return lib, os.path.basename(lib_path[0])

# version number
__version__ = libinfo.__version__
# library instance of TOPI
_LIB, _LIB_NAME = _load_lib()

#----------------------------
# helper function in ctypes.
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
        raise TVMError(py_str(_LIB.TVMGetLastError()))


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
