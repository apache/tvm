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
    keys = ["VTA_LOG_WGT_WIDTH",
            "VTA_LOG_INP_WIDTH",
            "VTA_LOG_ACC_WIDTH",
            "VTA_LOG_OUT_WIDTH",
            "VTA_LOG_BATCH",
            "VTA_LOG_BLOCK_IN",
            "VTA_LOG_BLOCK_OUT",
            "VTA_LOG_UOP_BUFF_SIZE",
            "VTA_LOG_INP_BUFF_SIZE",
            "VTA_LOG_WGT_BUFF_SIZE",
            "VTA_LOG_ACC_BUFF_SIZE",
            "VTA_LOG_OUT_BUFF_SIZE"]
    cflags = []
    for k in keys:
        cflags += ["-D%s=%s" % (k, str(getattr(env, k[4:])))]
    freconfig = remote.get_function("tvm.contrib.vta.reconfig_runtime")
    freconfig(" ".join(cflags))


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
