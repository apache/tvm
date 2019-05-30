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
"""Test ALU operations"""

######################################################################
# RPC Setup
# ---------
# We start by programming the Pynq's FPGA and building its RPC runtime
# as we did in the VTA introductory tutorial.

from __future__ import absolute_import, print_function

import argparse
import sys
import os
import tvm
import vta
import numpy as np
from tvm import rpc
from tvm.contrib import util
from vta.testing import simulator

def check_alu(tid, tvm_op, np_op=None, use_imm=False):
    """Test ALU"""

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

    m = 8
    n = 8
    imm = np.random.randint(1,5)

    # compute
    a = tvm.placeholder(
        (m, n, env.BATCH, env.BLOCK_OUT),
        name="a",
        dtype=env.acc_dtype)
    a_buf = tvm.compute(
        (m, n, env.BATCH, env.BLOCK_OUT),
        lambda *i: a(*i),
        "a_buf") #DRAM->SRAM
    if use_imm:
        res_buf = tvm.compute(
            (m, n, env.BATCH, env.BLOCK_OUT),
            lambda *i: tvm_op(a_buf(*i), imm),
            "res_buf") #compute
    else:
        b = tvm.placeholder(
            (m, n, env.BATCH, env.BLOCK_OUT),
            name="b",
            dtype=env.acc_dtype)
        b_buf = tvm.compute(
            (m, n, env.BATCH, env.BLOCK_OUT),
            lambda *i: b(*i),
            "b_buf") #DRAM->SRAM
        res_buf = tvm.compute(
            (m, n, env.BATCH, env.BLOCK_OUT),
            lambda *i: tvm_op(a_buf(*i), b_buf(*i)),
            "res_buf") #compute5B
    res = tvm.compute(
        (m, n, env.BATCH, env.BLOCK_OUT),
        lambda *i: res_buf(*i).astype(env.inp_dtype),
        "res") #SRAM->DRAM

    # schedule
    s = tvm.create_schedule(res.op)
    s[a_buf].set_scope(env.acc_scope) # SRAM
    s[a_buf].pragma(a_buf.op.axis[0], env.dma_copy) # DRAM->SRAM
    s[res_buf].set_scope(env.acc_scope) # SRAM
    s[res_buf].pragma(res_buf.op.axis[0], env.alu) # compute
    s[res].pragma(res.op.axis[0], env.dma_copy) # SRAM->DRAM
    if not use_imm:
        s[b_buf].set_scope(env.acc_scope) # SRAM
        s[b_buf].pragma(b_buf.op.axis[0], env.dma_copy) # DRAM->SRAM

    # build
    if use_imm:
        mod = vta.build(s, [a, res], "ext_dev", env.target_host)
    else:
        mod = vta.build(s, [a, b, res], "ext_dev", env.target_host)
    temp = util.tempdir()
    mod.save(temp.relpath("load_act.o"))
    remote.upload(temp.relpath("load_act.o"))
    f = remote.load_module("load_act.o")
    ctx = remote.ext_dev(0)

    # gen data
    a_np = np.random.randint(
        -16, 16, size=(m, n, env.BATCH, env.BLOCK_OUT)).astype(a.dtype)
    if use_imm:
        res_np = np_op(a_np, imm) if np_op else tvm_op(a_np, imm)
    else:
        b_np = np.random.randint(
            -16, 16, size=(m, n, env.BATCH, env.BLOCK_OUT)).astype(b.dtype)
        res_np = np_op(a_np, b_np) if np_op else tvm_op(a_np, b_np)
    res_np = res_np.astype(res.dtype)
    a_nd = tvm.nd.array(a_np, ctx)
    res_nd = tvm.nd.array(
        np.zeros((m, n, env.BATCH, env.BLOCK_OUT)).astype(res.dtype), ctx)

    # run
    if not use_imm:
        b_nd = tvm.nd.array(b_np, ctx)

    if use_imm:
        f(a_nd, res_nd)
    else:
        f(a_nd, b_nd, res_nd)

    # verify
    emsg = "[STATUS] FAIL " + __file__ + str(tid)
    np.testing.assert_equal(res_np, res_nd.asnumpy(), err_msg=emsg)
    print("[STATUS] PASS", tid, __file__)

def main():
    check_alu(0, lambda x, y: x << y, np.left_shift, use_imm=True)
    check_alu(1, tvm.max, np.maximum, use_imm=True)
    check_alu(2, tvm.max, np.maximum)
    check_alu(3, tvm.min, np.minimum, use_imm=True)
    check_alu(4, tvm.min, np.minimum)
    check_alu(5, lambda x, y: x + y, use_imm=True)
    check_alu(6, lambda x, y: x + y)
    check_alu(7, lambda x, y: x + y)
    check_alu(8, lambda x, y: x >> y, np.right_shift, use_imm=True)

if __name__ == "__main__":
    main()
