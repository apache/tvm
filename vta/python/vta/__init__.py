"""VTA Package is a TVM backend extension to support VTA hardwares

Besides the compiler toolchain.
It also include utility functions to
configure the hardware Environment and  access remote through RPC
"""
from __future__ import absolute_import as _abs

import sys

from .bitstream import get_bitstream_path, download_bitstream
from .environment import get_env, Environment
from .rpc_client import reconfig_runtime, program_fpga

__version__ = "0.1.0"

# do not import nnvm/topi when running vta.exec.rpc_server
# to maintain minimum dependency on the board
if sys.argv[0] not in ("-c", "-m"):
    from . import top
    from .build_module import build_config, lower, build
    from . import graph
