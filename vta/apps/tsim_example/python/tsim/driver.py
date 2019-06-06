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

def driver(hw_lib, sw_lib):
    """Init hardware and software shared library for add-by-one accelerator

     Parameters
     ------------
     hw_lib : str
        Name of hardware shared library

     sw_lib : str
        Name of software shared library
    """
    _cur_path = osp.dirname(osp.abspath(osp.expanduser(__file__)))
    _root_path = osp.join(_cur_path, "..", "..")
    _cfg_file = osp.join(_root_path, "config", "config.json")
    _cfg = json.load(open(_cfg_file))
    if not hw_lib.endswith(("dylib", "so")):
        hw_lib += ".dylib" if platform == "darwin" else ".so"
    if not sw_lib.endswith(("dylib", "so")):
        sw_lib += ".dylib" if platform == "darwin" else ".so"
    _hw_lib = osp.join(_root_path, _cfg['BUILD_NAME'], hw_lib)
    _sw_lib = osp.join(_root_path, _cfg['BUILD_NAME'], sw_lib)

    def load_dll(dll):
        try:
            return [ctypes.CDLL(dll, ctypes.RTLD_GLOBAL)]
        except OSError:
            return []

    def run(a, b):
        load_dll(_sw_lib)
        f = tvm.get_global_func("tvm.vta.driver")
        m = tvm.module.load(_hw_lib, "vta-tsim")
        f(m, a, b)
    return run
