# coding: utf-8
"""Information about nnvm."""
from __future__ import absolute_import
import sys
import os
import platform


def find_lib_path():
    """Find dynamic library files.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries
    """
    use_runtime = os.environ.get("TVM_USE_RUNTIME_LIB", False)
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    api_path = os.path.join(curr_path, '../../lib/')
    cmake_build_path = os.path.join(curr_path, '../../build/Release/')
    dll_path = [curr_path, api_path, cmake_build_path]
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
        lib_dll_path = [os.path.join(p, 'libtvm.dll') for p in dll_path]
        runtime_dll_path = [os.path.join(p, 'libtvm_runtime.dll') for p in dll_path]
    else:
        lib_dll_path = [os.path.join(p, 'libtvm.so') for p in dll_path]
        runtime_dll_path = [os.path.join(p, 'libtvm_runtime.so') for p in dll_path]

    dll_path = runtime_dll_path if use_runtime else lib_dll_path
    lib_found = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

    if len(lib_found) == 0:
        raise RuntimeError('Cannot find the files.\n' +
                           'List of candidates:\n' + str('\n'.join(dll_path)))
    if use_runtime:
        sys.stderr.write("Loading runtime library... this is execution only\n")
        sys.stderr.flush()
    return lib_found


# current version
__version__ = "0.1.0"
