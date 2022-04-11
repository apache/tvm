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

"""Testing topi gemm operator for VTA"""

import os
import json
from collections import namedtuple

import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm.contrib import utils
from tvm.contrib.pickle_memoize import memoize
from tvm import topi
import tvm.topi.testing
import vta
from vta import program_fpga, reconfig_runtime
import vta.testing
from vta.testing import simulator

# FIXME: we need a custom clip operator to circumvent a pattern detection limitation
@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x


def run_gemm(
    env,
    remote,
    target,
    batch_size,
    in_feat,
    out_feat,
    check_correctness=True,
    print_ir=True,
    samples=4,
):

    # Perform packing only if we are targeting the accelerator
    if "arm_cpu" in target.keys:
        data_pack = False
    elif "vta" in target.keys:
        data_pack = True

    # Derive shapes depending upon packing
    a_shape = (batch_size, in_feat)
    w_shape = (out_feat, in_feat)
    if data_pack:
        data_shape = (batch_size // env.BATCH, in_feat // env.BLOCK_IN, env.BATCH, env.BLOCK_IN)
        kernel_shape = (
            out_feat // env.BLOCK_OUT,
            in_feat // env.BLOCK_IN,
            env.BLOCK_OUT,
            env.BLOCK_IN,
        )
        fcompute = vta.top.dense_packed
        fschedule = vta.top.schedule_dense_packed
    else:
        data_shape = a_shape
        kernel_shape = w_shape
        fcompute = topi.x86.dense_nopack
        fschedule = topi.x86.schedule_dense_nopack
    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)

    # Define base computation schedule
    with target:
        res = fcompute(data, kernel, None, env.acc_dtype)
        res = topi.right_shift(res, 8)
        res = my_clip(res, 0, (1 << env.OUT_WIDTH - 1) - 1)
        res = topi.cast(res, env.out_dtype)
        # Derive base schedule
        s = fschedule([res])
        if print_ir:
            print(vta.lower(s, [data, kernel, res], simple_mode=True))

    # Derive number of ops
    num_ops = 2 * batch_size * in_feat * out_feat

    # @memoize("vta.tests.test_benchmark_topi.dense.verify")
    def get_ref_data():
        # derive min max for act, wgt types (max non inclusive)
        a_min, a_max = 0 - (1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
        w_min, w_max = 0 - (1 << (env.WGT_WIDTH - 1)), (1 << (env.WGT_WIDTH - 1))
        a_np = np.random.randint(a_min, a_max, size=a_shape).astype(data.dtype)
        w_np = np.random.randint(w_min, w_max, size=w_shape).astype(kernel.dtype)

        r_np = np.dot(a_np.astype(env.acc_dtype), w_np.T.astype(env.acc_dtype)).astype(
            env.acc_dtype
        )
        return a_np, w_np, r_np

    # Data in original format
    data_np, kernel_np, res_ref = get_ref_data()
    if data_pack:
        data_np = data_np.reshape(
            batch_size // env.BATCH, env.BATCH, in_feat // env.BLOCK_IN, env.BLOCK_IN
        ).transpose((0, 2, 1, 3))
        kernel_np = kernel_np.reshape(
            out_feat // env.BLOCK_OUT, env.BLOCK_OUT, in_feat // env.BLOCK_IN, env.BLOCK_IN
        ).transpose((0, 2, 1, 3))

    # Build
    if "vta" in target.keys:
        mod = vta.build(
            s,
            [data, kernel, res],
            target=tvm.target.Target(target, host=env.target_host),
            name="dense",
        )
    else:
        mod = tvm.build(
            s,
            [data, kernel, res],
            target=tvm.target.Target(target, host=env.target_host),
            name="dense",
        )
    temp = utils.tempdir()
    mod.save(temp.relpath("dense.o"))
    remote.upload(temp.relpath("dense.o"))
    f = remote.load_module("dense.o")
    dev = remote.device(str(target))

    res_np = np.zeros(topi.utils.get_const_tuple(res.shape)).astype(res.dtype)
    data_arr = tvm.nd.array(data_np, dev)
    kernel_arr = tvm.nd.array(kernel_np, dev)
    res_arr = tvm.nd.array(res_np, dev)
    time_f = f.time_evaluator("dense", dev, number=samples)

    # In vta sim mode, collect simulator runtime statistics
    stats = {}
    cost = None
    if env.TARGET in ["sim", "tsim"]:
        # Check if we're in local RPC mode (allows us to rebuild the
        # runtime on the fly when varying the VTA designs)
        local_rpc = int(os.environ.get("VTA_LOCAL_SIM_RPC", "0"))
        if local_rpc:
            if env.TARGET == "sim":
                remote.get_function("vta.simulator.profiler_clear")()
            else:
                remote.get_function("vta.tsim.profiler_clear")()
            cost = time_f(data_arr, kernel_arr, res_arr)
            if env.TARGET == "sim":
                stats = json.loads(remote.get_function("vta.simulator.profiler_status")())
            else:
                stats = json.loads(remote.get_function("vta.tsim.profiler_status")())
        else:
            simulator.clear_stats()
            cost = time_f(data_arr, kernel_arr, res_arr)
            stats = simulator.stats()
    else:
        cost = time_f(data_arr, kernel_arr, res_arr)

    # Check correctness
    correct = False
    if check_correctness:
        res_orig = res_arr.numpy()
        if data_pack:
            res_orig = res_orig.reshape(batch_size, out_feat)
        res_ref = res_ref >> 8
        res_ref = np.clip(res_ref, 0, (1 << env.OUT_WIDTH - 1) - 1)
        res_ref = res_ref.astype(env.out_dtype)
        correct = np.allclose(res_orig, res_ref)

    gops = (num_ops / cost.mean) / float(10**9)
    status = "PASSED" if correct else "FAILED"
    if "arm_cpu" in target.keys:
        device = "CPU"
    elif "vta" in target.keys:
        device = "VTA"
    print("%s DENSE TEST %s: Time cost = %g sec/op, %g GOPS" % (device, status, cost.mean, gops))

    return correct, cost, stats


def test_gemm(device="vta", batch=128, in_feat=128, out_feat=128):
    def _run(env, remote):
        if device == "vta":
            target = env.target
            if env.TARGET not in ["sim", "tsim"]:
                assert tvm.runtime.enabled("rpc")
                program_fpga(remote, bitstream=None)
                reconfig_runtime(remote)
        elif device == "arm_cpu":
            target = env.target_vta_cpu
        with autotvm.tophub.context(target):  # load pre-tuned schedule parameters
            run_gemm(env, remote, target, batch, in_feat, out_feat)

    vta.testing.run(_run)


if __name__ == "__main__":
    test_gemm("vta", 16, 512, 1008)
