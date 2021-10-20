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
"""VTA RPC client function"""
import os

from tvm import rpc
from vta import program_bitstream
from .environment import get_env
from .bitstream import download_bitstream, get_bitstream_path


def reconfig_runtime(remote):
    """Reconfigure remote runtime based on current hardware spec.

    Parameters
    ----------
    remote : RPCSession
        The TVM RPC session
    """
    env = get_env()
    freconfig = remote.get_function("tvm.contrib.vta.reconfig_runtime")
    freconfig(env.pkg.cfg_json)


def program_fpga(remote, bitstream=None):
    """Upload and program bistream

    Parameters
    ----------
    remote : RPCSession
        The TVM RPC session

    bitstream : str, optional
        Path to a local bistream file. If unset, tries to download from cache server.
    """
    env = get_env()

    if bitstream:
        assert os.path.isfile(bitstream)
    else:
        bitstream = get_bitstream_path()
        if not os.path.isfile(bitstream):
            if env.TARGET == "de10nano":
                return
            download_bitstream()

    if isinstance(remote, rpc.LocalSession):
        program_bitstream.bitstream_program(env.TARGET, bitstream)
    else:
        fprogram = remote.get_function("tvm.contrib.vta.init")
        remote.upload(bitstream)
        fprogram(os.path.basename(bitstream))
