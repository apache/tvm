import tvm
import vta
import os
from tvm.contrib import rpc, util

env = vta.get_env()
host = "pynq"
port = 9091
target = "llvm -target=armv7-none-linux-gnueabihf"
bit = "{}x{}x{}_{}bx{}b_{}_{}_{}_{}_100MHz_10ns.bit".format(
		env.BATCH, env.BLOCK_IN, env.BLOCK_OUT,
		env.INP_WIDTH, env.WGT_WIDTH,
		env.LOG_UOP_BUFF_SIZE, env.LOG_INP_BUFF_SIZE,
		env.LOG_WGT_BUFF_SIZE, env.LOG_ACC_BUFF_SIZE)

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
