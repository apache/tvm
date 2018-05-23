"""VTA RPC client function"""
import os

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
    freconfig(env.pkg_config().cfg_json)


def program_fpga(remote, bitstream=None):
    """Upload and program bistream

    Parameters
    ----------
    remote : RPCSession
        The TVM RPC session

    bitstream : str, optional
        Path to a local bistream file. If unset, tries to download from cache server.
    """
    if bitstream:
        assert os.path.isfile(bitstream)
    else:
        bitstream = get_bitstream_path()
        if not os.path.isfile(bitstream):
            download_bitstream()

    fprogram = remote.get_function("tvm.contrib.vta.init")
    remote.upload(bitstream)
    fprogram(os.path.basename(bitstream))
