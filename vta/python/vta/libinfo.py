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
"""Library information."""
from __future__ import absolute_import
import sys
import os

def _get_lib_name():
    if sys.platform.startswith('win32'):
        return "vta.dll"
    if sys.platform.startswith('darwin'):
        return "libvta.dylib"
    return "libvta.so"


def find_libvta(optional=False):
    """Find VTA library"""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_search = [curr_path]
    lib_search += [os.path.join(curr_path, "..", "..", "..", "build",)]
    lib_search += [os.path.join(curr_path, "..", "..", "..", "build", "Release")]
    lib_name = _get_lib_name()
    lib_path = [os.path.join(x, lib_name) for x in lib_search]
    lib_found = [x for x in lib_path if os.path.exists(x)]
    if not lib_found and not optional:
        raise RuntimeError('Cannot find the files.\n' +
                           'List of candidates:\n' + str('\n'.join(lib_path)))
    return lib_found
