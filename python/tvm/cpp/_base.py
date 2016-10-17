# coding: utf-8
# pylint: disable=invalid-name
""" ctypes library of nnvm and helper functions """
from __future__ import absolute_import

import sys
import os
import ctypes
import numpy as np
from . import libinfo

__all__ = ['TVMError']
#----------------------------
# library loading
#----------------------------
if sys.version_info[0] == 3:
    string_types = str,
    numeric_types = (float, int, np.float32, np.int32)
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    py_str = lambda x: x.decode('utf-8')
else:
    string_types = basestring,
    numeric_types = (float, int, long, np.float32, np.int32)
    py_str = lambda x: x


class TVMError(Exception):
    """Error that will be throwed by all functions"""
    pass

def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path()
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    # DMatrix functions
    lib.TVMGetLastError.restype = ctypes.c_char_p
    return lib

# version number
__version__ = libinfo.__version__
# library instance of nnvm
_LIB = _load_lib()

# type definitions
FunctionHandle = ctypes.c_void_p
NodeHandle = ctypes.c_void_p

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
        if len(arg_descs[i]) != 0:
            ret += '\n    ' + py_str(arg_descs[i])
        param_str.append(ret)
    doc_str = ('Parameters\n' +
               '----------\n' +
               '%s\n')
    doc_str = doc_str % ('\n'.join(param_str))
    return doc_str
