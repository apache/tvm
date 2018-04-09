import tvm
import vta
import os
from tvm.contrib import rpc, util

host = "pynq"
port = 9091
target = "llvm -target=armv7-none-linux-gnueabihf"
bit = "{}x{}x{}_{}bx{}b_{}_{}_{}_{}_100MHz_10ns.bit".format(
		vta.VTA_BATCH, vta.VTA_BLOCK_IN, vta.VTA_BLOCK_OUT,
		vta.VTA_INP_WIDTH, vta.VTA_WGT_WIDTH,
		vta.VTA_LOG_UOP_BUFF_SIZE, vta.VTA_LOG_INP_BUFF_SIZE,
		vta.VTA_LOG_WGT_BUFF_SIZE, vta.VTA_LOG_OUT_BUFF_SIZE)

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
bitstream = os.path.join(curr_path, "../../../../vta_bitstreams/bitstreams/", bit)

def test_program_rpc():
    assert tvm.module.enabled("rpc")
    remote = rpc.connect(host, port)
    vta.program_fpga(remote, bit)

def test_reconfig_runtime():
    assert tvm.module.enabled("rpc")
    remote = rpc.connect(host, port)
    vta.reconfig_runtime(remote)

test_program_rpc()
test_reconfig_runtime()
