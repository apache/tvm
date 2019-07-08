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
import tvm
from ..environment import get_env
from ..libinfo import find_libvta


def _load_sw():
    """Load software library, assuming they are simulator."""
    lib_sw = find_libvta("libvta", optional=True)
    if not lib_sw:
        return []
    try:
        return [ctypes.CDLL(lib_sw[0], ctypes.RTLD_GLOBAL)]
    except OSError:
        return []


def _load_all():
    """Load hardware library for tsim."""
    lib = _load_sw()
    env = get_env()
    if env.TARGET == "tsim":
        lib = find_libvta("libvta_hw", optional=True)
        f = tvm.get_global_func("vta.tsim.init")
        m = tvm.module.load(lib[0], "vta-tsim")
        f(m)
    return lib


def enabled():
    """Check if simulator is enabled."""
    f = tvm.get_global_func("vta.simulator.profiler_clear", True)
    return f is not None


def clear_stats():
    """Clear profiler statistics."""
    env = get_env()
    if env.TARGET == "sim":
        f = tvm.get_global_func("vta.simulator.profiler_clear", True)
    else:
        f = tvm.get_global_func("vta.tsim.profiler_clear", True)
    if f:
        f()


def stats():
    """Get profiler statistics

    Returns
    -------
    stats : dict
        Current profiler statistics
    """
    env = get_env()
    if env.TARGET == "sim":
        x = tvm.get_global_func("vta.simulator.profiler_status")()
    else:
        x = tvm.get_global_func("vta.tsim.profiler_status")()
    return json.loads(x)


# debug flag to skip execution.
DEBUG_SKIP_EXEC = 1

def debug_mode(flag):
    """Set debug mode
    Paramaters
    ----------
    flag : int
        The debug flag, 0 means clear all flags.
    """
    tvm.get_global_func("vta.simulator.profiler_debug_mode")(flag)


LIBS = _load_all()
