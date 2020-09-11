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
import os
import tvm
from tvm import te
from tvm import rpc
from vta import get_bitstream_path, download_bitstream, program_fpga, reconfig_runtime

host = os.environ.get("VTA_RPC_HOST", "pynq")
port = int(os.environ.get("VTA_RPC_PORT", "9091"))


def program_rpc_bitstream(path=None):
    """Program the FPGA on the RPC server

    Parameters
    ----------
    path : path to bitstream (optional)
    """
    assert tvm.runtime.enabled("rpc")
    remote = rpc.connect(host, port)
    program_fpga(remote, path)


def reconfig_rpc_runtime():
    """Reconfig the RPC server runtime"""
    assert tvm.runtime.enabled("rpc")
    remote = rpc.connect(host, port)
    reconfig_runtime(remote)


program_rpc_bitstream()
reconfig_rpc_runtime()
