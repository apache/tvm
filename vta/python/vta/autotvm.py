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

"""Defines AutoTVM components used with VTA."""

from tvm.autotvm.measure import default_module_loader
from . import rpc_client


def module_loader(bitstream=None):
    """Construct a ModuleLoader implementation specialized for VTA.

    Parameters
    ----------
    bitsream : Optional[str]
        Path to the bitstream to write prior to uploading code.

    Returns
    -------
    ModuleLoader :
        The ModuleLoader instance.
    """

    def reprogram_fpga(remote, _build_result):
        """default_module_loader callback which reprograms the FPGA.

        Parameters
        ----------
        remote : tvm.rpc.RPCSession
            RPC session established to the remote device.

        _build_result : tvm.autotvm.measure.measure_methods.BuildResult
            Artifact from the build phase, unused here.
        """
        rpc_client.program_fpga(remote, bitstream)
        rpc_client.reconfig_runtime(remote)

    return default_module_loader(reprogram_fpga)
