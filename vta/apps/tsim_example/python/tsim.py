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
import os.path as osp
from sys import platform

def driver(hw_backend):
    """Init hardware and software shared library for accelerator

     Parameters
     ------------
     hw_backend : str
        Hardware backend can be verilog or chisel

    """
    _ext = ".dylib" if platform == "darwin" else ".so"
    _hw_libname = "libhw" + _ext
    _sw_libname = "libsw" + _ext
    _cur_path = osp.dirname(osp.abspath(osp.expanduser(__file__)))
    if hw_backend in ("verilog", "chisel"):
        _hw_lib = osp.join(_cur_path, "..", "hardware", hw_backend, "build", _hw_libname)
    _sw_lib = osp.join(_cur_path, "..", "build", _sw_libname)

    def load_dll(dll):
        try:
            return [ctypes.CDLL(dll, ctypes.RTLD_GLOBAL)]
        except OSError:
            return []

    def run(a, b, c):
        load_dll(_sw_lib)
        f = tvm.get_global_func("tvm.vta.driver")
        m = tvm.module.load(_hw_lib, "vta-tsim")
        return f(m, a, b, c)
    return run
