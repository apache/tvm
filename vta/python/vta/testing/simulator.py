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
# pylint: disable=superfluous-parens
"""Utilities to start simulator."""
import ctypes
import json
import warnings
import tvm
from ..environment import get_env
from ..libinfo import find_libvta


def _load_sw():
    """Load hardware library for simulator."""

    env = get_env()
    lib_driver_name = (
        "libvta_tsim"
        if env.TARGET == "tsim"
        else "libvta"
        if env.TARGET == "intelfocl"
        else "libvta_fsim"
    )
    require_sim = env.TARGET in ("sim", "tsim")
    libs = []

    # Load driver library
    lib_driver = find_libvta(lib_driver_name, optional=(not require_sim))

    if not lib_driver:
        return []

    try:
        libs = [ctypes.CDLL(lib_driver[0], ctypes.RTLD_GLOBAL)]
    except OSError as err:
        if require_sim:
            raise err
        warnings.warn("Error when loading VTA driver {}: {}".format(lib_driver[0], err))
        return []

    if env.TARGET == "tsim":
        lib_hw = find_libvta("libvta_hw", optional=True)
        assert lib_hw  # make sure to make in ${VTA_HW_PATH}/hardware/chisel
        f = tvm.get_global_func("vta.tsim.init")
        m = tvm.runtime.load_module(lib_hw[0], "vta-tsim")
        f(m)
        return lib_hw

    return libs


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


LIBS = _load_sw()
