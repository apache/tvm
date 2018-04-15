"""TVM-based VTA Compiler Toolchain"""
from __future__ import absolute_import as _abs

__version__ = "0.1.0"


from .environment import get_env, Environment
from .rpc_client import reconfig_runtime, program_fpga


try:
    from . import top
    from .build_module import build_config, lower, build
    from . import graph
except (ImportError, RuntimeError):
    pass
