import tvm
import vta
import os
from tvm.contrib import rpc, util

host = "pynq"
port = 9091
target = "llvm -target=armv7-none-linux-gnueabihf"
bit = "vta.bit"

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
bitstream = os.path.join(curr_path, "./", bit)

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
