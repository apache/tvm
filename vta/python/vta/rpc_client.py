"""VTA RPC client function"""
import os

from .environment import get_env

def reconfig_runtime(remote):
    """Reconfigure remote runtime based on current hardware spec.

    Parameters
    ----------
    remote : RPCSession
        The TVM RPC session
    """
    env = get_env()
    freconfig = remote.get_function("tvm.contrib.vta.reconfig_runtime")
    freconfig(env.pkg_config().cfg_json)


def program_fpga(remote, bitstream):
    """Upload and program bistream

    Parameters
    ----------
    remote : RPCSession
        The TVM RPC session

    bitstream : str
        Path to a local bistream file.
    """
    fprogram = remote.get_function("tvm.contrib.vta.init")
    remote.upload(bitstream)
    fprogram(os.path.basename(bitstream))
