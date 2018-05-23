import os
import tvm
from tvm.contrib import rpc
from vta import get_bitstream_path, download_bitstream, program_fpga, reconfig_runtime

def program_rpc_bitstream(path=None):
    """Program the FPGA on the RPC server

    Parameters
    ----------
    path : path to bitstream (optional)
    """
    assert tvm.module.enabled("rpc")
    host = os.environ.get("VTA_PYNQ_RPC_HOST", None)
    if not host:
        raise RuntimeError(
            "Error: VTA_PYNQ_RPC_HOST environment variable not set.")
    # If a path to a bitstream is passed, make sure that it point to a valid bitstream
    port = os.environ.get("VTA_PYNQ_RPC_PORT", "9091")
    port = int(port)
    remote = rpc.connect(host, port)
    program_fpga(remote, path)

def reconfig_rpc_runtime():
    """Reconfig the RPC server runtime
    """
    assert tvm.module.enabled("rpc")
    host = os.environ.get("VTA_PYNQ_RPC_HOST", None)
    if host:
        port = os.environ.get("VTA_PYNQ_RPC_PORT", "9091")
        port = int(port)
        remote = rpc.connect(host, port)
        reconfig_runtime(remote)

program_rpc_bitstream()
reconfig_rpc_runtime()
