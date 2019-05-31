# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test relu"""

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
elif env.TARGET in ["sim", "tsim"]:
    remote = rpc.LocalSession()

if env.TARGET == "tsim":
    simulator.tsim_init("libvta_hw")

m = 8
n = 10

# compute
a = tvm.placeholder(
    (m, n, env.BATCH, env.BLOCK_OUT),
    name="a",
    dtype=env.acc_dtype)
a_buf = tvm.compute(
    (m, n, env.BATCH, env.BLOCK_OUT),
    lambda *i: a(*i),
    "a_buf") # DRAM->SRAM
max_buf = tvm.compute(
    (m, n, env.BATCH, env.BLOCK_OUT),
    lambda *i: tvm.max(a_buf(*i), 0),
    "res_buf") # relu
min_buf = tvm.compute(
    (m, n, env.BATCH, env.BLOCK_OUT),
    lambda *i: tvm.min(max_buf(*i), (1<<(env.INP_WIDTH-1))-1),
    "max_buf") # relu
res = tvm.compute(
    (m, n, env.BATCH, env.BLOCK_OUT),
    lambda *i: min_buf(*i).astype(env.inp_dtype),
    "min_buf") # SRAM->DRAM

# schedule
s = tvm.create_schedule(res.op)
s[a_buf].set_scope(env.acc_scope) # SRAM
s[a_buf].pragma(a_buf.op.axis[0], env.dma_copy) # DRAM->SRAM
s[max_buf].set_scope(env.acc_scope) # SRAM
s[min_buf].set_scope(env.acc_scope) # SRAM
s[max_buf].pragma(max_buf.op.axis[0], env.alu) # compute
s[min_buf].pragma(min_buf.op.axis[0], env.alu) # compute
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
    -256, 256, size=(m, n, env.BATCH, env.BLOCK_OUT)).astype(a.dtype)
res_np = np.clip(a_np, 0, (1<<(env.INP_WIDTH-1))-1).astype(res.dtype)
a_nd = tvm.nd.array(a_np, ctx)
res_nd = tvm.nd.array(
    np.zeros((m, n, env.BATCH, env.BLOCK_OUT)).astype(res.dtype), ctx)

# run
f(a_nd, res_nd)

# verify
emsg = "[STATUS] FAIL " + __file__
np.testing.assert_equal(res_np, res_nd.asnumpy(), err_msg=emsg)
print("[STATUS] PASS", __file__)
