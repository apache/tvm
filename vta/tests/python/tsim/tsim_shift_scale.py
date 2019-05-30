"""
Shift and scale test
=======================
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

m = 2
n = 8

imm_shift = np.random.randint(0,8)
imm_scale = np.random.randint(1,5)

# compute
a = tvm.placeholder(
    (m, n, env.BATCH, env.BLOCK_OUT),
    name="a", dtype=env.acc_dtype)
a_buf = tvm.compute(
    (m, n, env.BATCH, env.BLOCK_OUT),
    lambda *i: a(*i),
    "a_buf") # DRAM->SRAM
res_shift = tvm.compute(
    (m, n, env.BATCH, env.BLOCK_OUT),
    lambda *i: a_buf(*i)+imm_shift,
    "res_shift") # compute
res_scale = tvm.compute(
    (m, n, env.BATCH, env.BLOCK_OUT),
    lambda *i: res_shift(*i)>>imm_scale,
    "res_scale") # compute
res = tvm.compute(
    (m, n, env.BATCH, env.BLOCK_OUT),
    lambda *i: res_scale(*i).astype(env.inp_dtype),
    "res") # SRAM->DRAM

# schedule
s = tvm.create_schedule(res.op)
s[a_buf].set_scope(env.acc_scope) # SRAM
s[res_shift].set_scope(env.acc_scope) # SRAM
s[res_scale].set_scope(env.acc_scope) # SRAM
s[a_buf].pragma(a_buf.op.axis[0], env.dma_copy) # DRAM->SRAM
s[res_shift].pragma(res_shift.op.axis[0], env.alu) # compute
s[res_scale].pragma(res_scale.op.axis[0], env.alu) # compute
s[res].pragma(res.op.axis[0], env.dma_copy) # SRAM->DRAM

# build
mod = vta.build(s, [a, res], "ext_dev", env.target_host)
temp = util.tempdir()
mod.save(temp.relpath("load_act.o"))
remote.upload(temp.relpath("load_act.o"))
f = remote.load_module("load_act.o")
ctx = remote.ext_dev(0)

# gen data
a_np = np.random.randint(
    -10, 10, size=(m, n, env.BATCH, env.BLOCK_OUT)).astype(a.dtype)
res_np = np.right_shift((a_np + imm_shift), imm_scale)
res_np = res_np.astype(res.dtype)
a_nd = tvm.nd.array(a_np, ctx)
res_nd = tvm.nd.array(
    np.zeros((m, n, env.BATCH, env.BLOCK_OUT)).astype(res.dtype), ctx)

# run
f(a_nd, res_nd)

# verify
emsg = "[STATUS] FAIL " + __file__
np.testing.assert_equal(res_np, res_nd.asnumpy(), err_msg=emsg)
print("[STATUS] PASS", __file__)
