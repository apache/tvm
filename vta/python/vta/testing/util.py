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
"""Test Utilities"""
from __future__ import absolute_import as _abs

import os
from tvm import rpc
from ..environment import get_env
from . import simulator

def run(run_func):
    """Run test function on all available env.

    Parameters
    ----------
    run_func : function(env, remote)
    """
    env = get_env()

    if env.TARGET in ["sim", "tsim"]:

        # Talk to local RPC if necessary to debug RPC server.
        # Compile vta on your host with make at the root.
        # Make sure TARGET is set to "sim" in the config.json file.
        # Then launch the RPC server on the host machine
        # with ./apps/pynq_rpc/start_rpc_server.sh
        # Set your VTA_LOCAL_SIM_RPC environment variable to
        # the port it's listening to, e.g. 9090
        local_rpc = int(os.environ.get("VTA_LOCAL_SIM_RPC", "0"))
        if local_rpc:
            remote = rpc.connect("localhost", local_rpc)
            run_func(env, remote)
        else:
            # Make sure simulation library exists
            # If this fails, build vta on host (make)
            # with TARGET="sim" in the json.config file.
            if env.TARGET == "sim":
                assert simulator.enabled()
            run_func(env, rpc.LocalSession())

    elif env.TARGET == "pynq":

        # Run on PYNQ if env variable exists
        host = os.environ.get("VTA_PYNQ_RPC_HOST", None)
        port = int(os.environ.get("VTA_PYNQ_RPC_PORT", None))
        if host and port:
            remote = rpc.connect(host, port)
            run_func(env, remote)
        else:
            raise RuntimeError(
                "Please set the VTA_PYNQ_RPC_HOST and VTA_PYNQ_RPC_PORT environment variables")
