"""Test Utilities"""
from __future__ import absolute_import as _abs

import os
from tvm.contrib import rpc
from ..environment import get_env
from . import simulator


def run(run_func):
    """Run test function on all available env.

    Parameters
    ----------
    run_func : function(env, remote)
    """
    env = get_env()

    # Run on local sim rpc if necessary
    local_rpc = int(os.environ.get("VTA_LOCAL_SIM_RPC", "0"))
    if local_rpc:
        env.TARGET = "sim"
        remote = rpc.connect("localhost", local_rpc)
        run_func(env, remote)
    else:
        # run on simulator
        if simulator.enabled():
            env.TARGET = "sim"
            run_func(env, rpc.LocalSession())

    # Run on PYNQ if env variable exists
    pynq_host = os.environ.get("VTA_PYNQ_RPC_HOST", None)
    if pynq_host:
        env.TARGET = "pynq"
        port = os.environ.get("VTA_PYNQ_RPC_PORT", "9091")
        port = int(port)
        remote = rpc.connect(pynq_host, port)
        run_func(env, remote)
