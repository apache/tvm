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

def get_ext():
    """Return shared library extension"""
    return ".dylib" if platform == "darwin" else ".so"

def load_dll(dll):
    """Load shared library

    Parameters
    ------------
    dll : str
        Path for shared library

    Returns
    ------------
    The shared library
    """
    try:
        return [ctypes.CDLL(dll, ctypes.RTLD_GLOBAL)]
    except OSError:
        return []

def load_sw():
    """Load all software shared libraries"""
    cur_path = osp.dirname(osp.abspath(osp.expanduser(__file__)))
    sw_libname = "libsw" + get_ext()
    sw_lib = osp.join(cur_path, "..", "build", sw_libname)
    load_dll(sw_lib)

def init(hw_backend):
    """Init hardware and software shared library for accelerator

    Parameters
    ------------
    hw_backend : str
        Hardware backend can be verilog or chisel

    """
    cur_path = osp.dirname(osp.abspath(osp.expanduser(__file__)))
    hw_libname = "libhw" + get_ext()
    if hw_backend in ("verilog", "chisel"):
        hw_lib = osp.join(cur_path, "..", "hardware", hw_backend, "build", hw_libname)
    load_sw()
    m = tvm.module.load(hw_lib, "vta-tsim")
    f = tvm.get_global_func("tvm.vta.tsim.init")
    f(m)

def load_module():
    """Return driver function"""
    load_sw()
    return tvm.get_global_func("tvm.vta.driver")
