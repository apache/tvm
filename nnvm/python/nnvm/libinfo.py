# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
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
        lib_name = "nnvm_compiler" if sys.platform.startswith('win32') else "libnnvm_compiler"

    api_path = os.path.join(base_path, '..', '..', 'lib')
    cmake_build_path_win = os.path.join(base_path, '..', '..', '..', 'build', 'Release')
    cmake_build_path = os.path.join(base_path, '..', '..', '..', 'build')
    install_path = os.path.join(base_path, '..', '..', '..')
    dll_path = [base_path, api_path, cmake_build_path_win, cmake_build_path,
                install_path]

    if sys.platform.startswith('linux') and os.environ.get('LD_LIBRARY_PATH', None):
        dll_path.extend([p.strip() for p in os.environ['LD_LIBRARY_PATH'].split(":")])
    elif sys.platform.startswith('darwin') and os.environ.get('DYLD_LIBRARY_PATH', None):
        dll_path.extend([p.strip() for p in os.environ['DYLD_LIBRARY_PATH'].split(":")])
    elif sys.platform.startswith('win32') and os.environ.get('PATH', None):
        dll_path.extend([p.strip() for p in os.environ['PATH'].split(";")])

    if sys.platform.startswith('win32'):
        vs_configuration = 'Release'
        if platform.architecture()[0] == '64bit':
            dll_path.append(os.path.join(base_path, '..', '..', '..', 'build', vs_configuration))
            dll_path.append(os.path.join(base_path, '..', '..', '..', 'windows', 'x64',
                                         vs_configuration))
        else:
            dll_path.append(os.path.join(base_path, '..', '..', '..', 'build', vs_configuration))
            dll_path.append(os.path.join(base_path, '..', '..', '..', 'windows', vs_configuration))
        dll_path = [os.path.join(p, '%s.dll' % lib_name) for p in dll_path]
    elif sys.platform.startswith('darwin'):
        dll_path = [os.path.join(p, '%s.dylib' % lib_name) for p in dll_path]
    else:
        dll_path = [os.path.join(p, '%s.so' % lib_name) for p in dll_path]

    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if not lib_path:
        raise RuntimeError('Cannot find the files.\n' +
                           'List of candidates:\n' + str('\n'.join(dll_path)))
    return lib_path


# current version
__version__ = "0.8.0"
