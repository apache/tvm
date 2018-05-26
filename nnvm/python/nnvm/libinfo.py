# coding: utf-8
"""Information about nnvm."""
from __future__ import absolute_import
import sys
import os
import platform

if sys.version_info[0] == 3:
    import builtins as __builtin__
else:
    import __builtin__

def find_lib_path():
    """Find NNNet dynamic library files.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries
    """
    if hasattr(__builtin__, "NNVM_BASE_PATH"):
        base_path = __builtin__.NNVM_BASE_PATH
    else:
        base_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

    if hasattr(__builtin__, "NNVM_LIBRARY_NAME"):
        lib_name = __builtin__.NNVM_LIBRARY_NAME
    else:
        lib_name = "libnnvm_example"

    api_path = os.path.join(base_path, '../../lib/')
    cmake_build_path = os.path.join(base_path, '../../build/Release/')
    dll_path = [base_path, api_path, cmake_build_path]
    if os.name == 'nt':
        vs_configuration = 'Release'
        if platform.architecture()[0] == '64bit':
            dll_path.append(os.path.join(curr_path, '../../build', vs_configuration))
            dll_path.append(os.path.join(curr_path, '../../windows/x64', vs_configuration))
        else:
            dll_path.append(os.path.join(curr_path, '../../build', vs_configuration))
            dll_path.append(os.path.join(curr_path, '../../windows', vs_configuration))
    elif os.name == "posix" and os.environ.get('LD_LIBRARY_PATH', None):
        dll_path.extend([p.strip() for p in os.environ['LD_LIBRARY_PATH'].split(":")])
    if os.name == 'nt':
        dll_path = [os.path.join(p, '%s.dll' % lib_name) for p in dll_path]
    else:
        dll_path = [os.path.join(p, '%s.so' % lib_name) for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if len(lib_path) == 0:
        raise RuntimeError('Cannot find the files.\n' +
                           'List of candidates:\n' + str('\n'.join(dll_path)))
    return lib_path


# current version
__version__ = "0.7.0"
