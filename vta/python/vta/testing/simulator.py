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
"""Utilities to start simulator."""
import ctypes
import json
import sys
import os
import tvm
from ..libinfo import find_libvta

def _load_lib():
    """Load local library, assuming they are simulator."""
    lib_path = find_libvta(optional=True)
    if not lib_path:
        return []
    try:
        return [ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)]
    except OSError:
        return []


def enabled():
    """Check if simulator is enabled."""
    f = tvm.get_global_func("vta.simulator.profiler_clear", True)
    return f is not None


def clear_stats():
    """Clear profiler statistics"""
    f = tvm.get_global_func("vta.simulator.profiler_clear", True)
    if f:
        f()


def stats():
    """Clear profiler statistics

    Returns
    -------
    stats : dict
        Current profiler statistics
    """
    x = tvm.get_global_func("vta.simulator.profiler_status")()
    return json.loads(x)

def tsim_init(hw_lib):
    """Init hardware shared library for TSIM

     Parameters
     ------------
     hw_lib : str
        Name of hardware shared library
    """
    cur_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    vta_build_path = os.path.join(cur_path, "..", "..", "..", "build")
    if not hw_lib.endswith(("dylib", "so")):
        hw_lib += ".dylib" if sys.platform == "darwin" else ".so"
    lib = os.path.join(vta_build_path, hw_lib)
    f = tvm.get_global_func("tvm.vta.tsim.init")
    m = tvm.module.load(lib, "vta-tsim")
    f(m)


LIBS = _load_lib()
