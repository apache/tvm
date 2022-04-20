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

"""Testing topi conv2d_transpose operator for VTA"""

import json
import os

import pytest
import numpy as np
from collections import namedtuple

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from tvm.contrib import utils
from tvm.contrib.pickle_memoize import memoize
from tvm import topi
import tvm.topi.testing
import vta
from vta import program_fpga, reconfig_runtime
import vta.testing
from vta.testing import simulator


Workload = namedtuple(
    "Conv2DTransposeWorkload",
    [
        "batch",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
        "o_hpad",
        "o_wpad",
    ],
)

# Get batch info from env
env = vta.get_env()

# DCGAN workloads
dcgan_wklds = [
    # dcgan
    ("DCGAN.CT1", Workload(env.BATCH, 4, 4, 1024, 512, 4, 4, 1, 1, 2, 2, 0, 0)),
    ("DCGAN.CT2", Workload(env.BATCH, 8, 8, 512, 256, 4, 4, 1, 1, 2, 2, 0, 0)),
    ("DCGAN.CT3", Workload(env.BATCH, 16, 16, 256, 128, 4, 4, 1, 1, 2, 2, 0, 0)),
]

# FIXME: we need a custom clip operator to circumvent a pattern detection limitation
@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x


# Helper function to get factors
def _find_factors(n):
    factors = []
    for f in range(1, n + 1):
        if n % f == 0:
            factors.append(f)
    return factors


def run_conv2d_transpose(
    env, remote, wl, target, check_correctness=True, print_ir=False, samples=4
):

    # Workload assertions
    assert wl.hpad == wl.wpad

    # Perform packing only if we are targeting the accelerator
    if "arm_cpu" in target.keys:
        data_pack = False
        layout = "NCHW"
        fcompute = topi.arm_cpu.conv2d_transpose_nchw
        fschedule = topi.arm_cpu.schedule_conv2d_transpose_nchw
    elif "vta" in target.keys:
        data_pack = True
        layout = "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN)
        fcompute = vta.top.conv2d_transpose_packed
        fschedule = vta.top.schedule_conv2d_transpose_packed

    # Derive shapes depending upon packing

    a_shape = (wl.batch, wl.in_filter, wl.height, wl.width)
    w_shape = (wl.in_filter, wl.out_filter, wl.hkernel, wl.wkernel)
    if data_pack:
        data_shape = (
            wl.batch // env.BATCH,
            wl.in_filter // env.BLOCK_IN,
            wl.height,
            wl.width,
            env.BATCH,
            env.BLOCK_IN,
        )
        kernel_shape = (
            wl.out_filter // env.BLOCK_OUT,
            wl.in_filter // env.BLOCK_IN,
            wl.hkernel,
            wl.wkernel,
            env.BLOCK_OUT,
            env.BLOCK_IN,
        )
    else:
        data_shape = a_shape
        kernel_shape = w_shape
    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
    padding = relay.nn.get_pad_tuple2d((wl.hpad, wl.wpad))

    # Define base computation schedule
    with target:

        res = fcompute(
            data, kernel, (wl.hstride, wl.wstride), padding, env.acc_dtype, (wl.o_hpad, wl.o_wpad)
        )
        res = topi.right_shift(res, env.WGT_WIDTH)
        res = my_clip(res, 0, (1 << env.OUT_WIDTH - 1) - 1)
        res = topi.cast(res, env.out_dtype)
        # Derive base schedule
        s = fschedule([res])
        if print_ir:
            print(vta.lower(s, [data, kernel, res], simple_mode=True))

    # Derive number of ops
    fout_height = (wl.height - 1) * wl.hstride - 2 * wl.hpad + wl.hkernel + wl.o_hpad
    fout_width = (wl.width - 1) * wl.wstride - 2 * wl.wpad + wl.wkernel + wl.o_wpad
    num_ops = (
        2
        * wl.batch
        * fout_height
        * fout_width
        * wl.hkernel
        * wl.wkernel
        * wl.out_filter
        * wl.in_filter
    )

    # @memoize("vta.tests.test_benchmark_topi.conv2d.verify_nhwc")
    def get_ref_data():
        # derive min max for act and wgt types (max non inclusive)
        a_min, a_max = 0 - (1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
        w_min, w_max = 0 - (1 << (env.WGT_WIDTH - 1)), (1 << (env.WGT_WIDTH - 1))
        a_np = np.random.randint(a_min, a_max, size=a_shape).astype(data.dtype)
        w_np = np.random.randint(
            w_min, w_max, size=(wl.in_filter, wl.out_filter, wl.hkernel, wl.wkernel)
        ).astype(kernel.dtype)
        r_np = tvm.topi.testing.conv2d_transpose_nchw_python(
            a_np.astype(env.acc_dtype),
            w_np.astype(env.acc_dtype),
            (wl.hstride, wl.wstride),
            wl.hpad,
            (wl.o_hpad, wl.o_wpad),
        ).astype(env.acc_dtype)
        return a_np, w_np, r_np

    # Data in original format
    data_np, kernel_np, res_ref = get_ref_data()
    if data_pack:
        data_np = data_np.reshape(
            wl.batch // env.BATCH,
            env.BATCH,
            wl.in_filter // env.BLOCK_IN,
            env.BLOCK_IN,
            wl.height,
            wl.width,
        ).transpose((0, 2, 4, 5, 1, 3))
        kernel_np = kernel_np.reshape(
            wl.in_filter // env.BLOCK_IN,
            env.BLOCK_IN,
            wl.out_filter // env.BLOCK_OUT,
            env.BLOCK_OUT,
            wl.hkernel,
            wl.wkernel,
        ).transpose((2, 0, 4, 5, 3, 1))
        kernel_np = np.flip(kernel_np, 2)
        kernel_np = np.flip(kernel_np, 3)

    # Build
    if "vta" in target.keys:
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            mod = vta.build(
                s,
                [data, kernel, res],
                target=target,
                target_host=env.target_host,
                name="conv2d_transpose",
            )
    else:
        mod = tvm.build(
            s,
            [data, kernel, res],
            target=target,
            target_host=env.target_host,
            name="conv2d_transpose",
        )
    temp = utils.tempdir()
    mod.save(temp.relpath("conv2d_transpose.o"))
    remote.upload(temp.relpath("conv2d_transpose.o"))
    f = remote.load_module("conv2d_transpose.o")
    dev = remote.device(str(target))

    res_np = np.zeros(topi.utils.get_const_tuple(res.shape)).astype(res.dtype)
    data_arr = tvm.nd.array(data_np, dev)
    kernel_arr = tvm.nd.array(kernel_np, dev)
    res_arr = tvm.nd.array(res_np, dev)
    time_f = f.time_evaluator("conv2d_transpose", dev, number=samples)

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
            res_orig = res_orig.transpose((0, 4, 1, 5, 2, 3)).reshape(
                wl.batch, wl.out_filter, fout_height, fout_width
            )
        res_ref = res_ref >> env.WGT_WIDTH
        res_ref = np.clip(res_ref, 0, (1 << env.OUT_WIDTH - 1) - 1)
        res_ref = res_ref.astype(env.out_dtype)
        correct = np.allclose(res_orig, res_ref)

    gops = (num_ops / cost.mean) / float(10**9)
    status = "PASSED" if correct else "FAILED"
    if "arm_cpu" in target.keys:
        device = "CPU"
    elif "vta" in target.keys:
        device = "VTA"
    print("%s CONV2D TEST %s: Time cost = %g sec/op, %g GOPS" % (device, status, cost.mean, gops))

    return correct, cost, stats


@pytest.mark.parametrize("device", ["vta", "arm_cpu"])
def test_conv2d_transpose(device):
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
            for _, wl in dcgan_wklds:
                print(wl)
                run_conv2d_transpose(env, remote, wl, target)

    vta.testing.run(_run)


if __name__ == "__main__":
    test_conv2d_transpose(device="arm_cpu")
    test_conv2d_transpose(device="vta")
