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

"""VTA Package is a TVM backend extension to support VTA hardware.

Besides the compiler toolchain, it also includes utility functions to
configure the hardware environment and access remote device through RPC.
"""
import sys
import tvm._ffi.base

from .autotvm import module_loader
from .bitstream import get_bitstream_path, download_bitstream
from .environment import get_env, Environment
from .rpc_client import reconfig_runtime, program_fpga

__version__ = "0.1.0"


# do not from tvm import topi when running vta.exec.rpc_server
# in lib tvm runtime only mode
if not tvm._ffi.base._RUNTIME_ONLY:
    from . import top
    from .build_module import build_config, lower, build
