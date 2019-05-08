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

import tvm
import ctypes
import json
import os.path as osp
from sys import platform

def get_build_path():
    curr_path = osp.dirname(osp.abspath(osp.expanduser(__file__)))
    cfg = json.load(open(osp.join(curr_path, 'config.json')))
    return osp.join(curr_path, "..", "..", cfg['BUILD_NAME'])

def get_lib_ext():
    if platform == "darwin":
        ext = ".dylib"
    else:
        ext = ".so"
    return ext

def get_lib_path(name):
    build_path = get_build_path()
    ext = get_lib_ext()
    libname = name + ext
    return osp.join(build_path, libname)

def _load_driver_lib():
    lib = get_lib_path("libdriver")
    try:
        return [ctypes.CDLL(lib, ctypes.RTLD_GLOBAL)]
    except OSError:
        return []

def load_driver():
    return tvm.get_global_func("tvm.vta.driver")

def load_tsim():
    lib = get_lib_path("libtsim")
    return tvm.module.load(lib, "vta-tsim")

LIBS = _load_driver_lib()
