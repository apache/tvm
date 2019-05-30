"""
Load and store test
======================
"""

######################################################################
# RPC Setup
# ---------
# We start by programming the Pynq's FPGA and building its RPC runtime
# as we did in the VTA introductory tutorial.

from __future__ import absolute_import, print_function

import os
import tvm
import vta
import numpy as np
from tvm import rpc
from tvm.contrib import util
from vta.testing import simulator

# Load VTA parameters from the vta/config/vta_config.json file
env = vta.get_env()

# We read the Pynq RPC host IP address and port number from the OS environment
host = os.environ.get("VTA_PYNQ_RPC_HOST", "192.168.2.99")
port = int(os.environ.get("VTA_PYNQ_RPC_PORT", "9091"))

# We configure both the bitstream and the runtime system on the Pynq
# to match the VTA configuration specified by the vta_config.json file.
if env.TARGET == "pynq":

    # Make sure that TVM was compiled with RPC=1
    assert tvm.module.enabled("rpc")
    remote = rpc.connect(host, port)

    # Reconfigure the JIT runtime
    vta.reconfig_runtime(remote)

    # Program the FPGA with a pre-compiled VTA bitstream.
    # You can program the FPGA with your own custom bitstream
    # by passing the path to the bitstream file instead of None.
    vta.program_fpga(remote, bitstream=None)

# In simulation mode, host the RPC server locally.
elif env.TARGET == "sim" or env.TARGET == "tsim":
    remote = rpc.LocalSession()

if env.TARGET == "tsim":
    simulator.tsim_init("libvta_hw")

n = 10

# compute
x = tvm.placeholder(
    (n, n, env.BATCH, env.BLOCK_OUT),
    name="x",
    dtype=env.acc_dtype)
x_buf = tvm.compute(
    (n, n, env.BATCH, env.BLOCK_OUT),
    lambda *i: x(*i), "x_buf")
# insert no-op that won't be optimized away
y_buf = tvm.compute(
    (n, n, env.BATCH, env.BLOCK_OUT),
    lambda *i: x_buf(*i)>>0, "y_buf")
y = tvm.compute(
    (n, n, env.BATCH, env.BLOCK_OUT),
    lambda *i: y_buf(*i).astype(env.inp_dtype), "y")

# schedule
s = tvm.create_schedule(y.op)
s[x_buf].set_scope(env.acc_scope)
s[x_buf].pragma(x_buf.op.axis[0], env.dma_copy)
s[y_buf].set_scope(env.acc_scope)
s[y_buf].pragma(y_buf.op.axis[0], env.alu)
s[y].pragma(y.op.axis[0], env.dma_copy)

# build
mod = vta.build(s, [x, y], "ext_dev", env.target_host)
temp = util.tempdir()
mod.save(temp.relpath("load_act.o"))
remote.upload(temp.relpath("load_act.o"))
f = remote.load_module("load_act.o")
ctx = remote.ext_dev(0)

# gen data
x_np = np.random.randint(
    1, 10, size=(n, n, env.BATCH, env.BLOCK_OUT)).astype(x.dtype)
y_np = x_np.astype(y.dtype)
x_nd = tvm.nd.array(x_np, ctx)
y_nd = tvm.nd.empty(y_np.shape, ctx=ctx, dtype=y_np.dtype)

# run
f(x_nd, y_nd)

# verify
emsg = "[STATUS] FAIL " + __file__
np.testing.assert_equal(y_np, y_nd.asnumpy(), err_msg=emsg)
print("[STATUS] PASS", __file__)
