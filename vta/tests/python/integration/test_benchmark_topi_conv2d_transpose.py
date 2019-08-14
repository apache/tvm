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

import numpy as np
from collections import namedtuple

import tvm
from tvm import autotvm
from tvm.contrib import util
from tvm.contrib.pickle_memoize import memoize
import topi
import topi.testing
import vta
from vta import program_fpga, reconfig_runtime
import vta.testing
from vta.testing import simulator


Workload = namedtuple("Conv2DTransposeWorkload",
                      ['batch', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

# Get batch info from env
env = vta.get_env()

# DCGAN workloads
dcgan_wklds = [
    # dcgan
    ('DCGAN.CT1', Workload(env.BATCH,  4,  4, 1024, 512, 4, 4, 1, 1, 2, 2)),
    ('DCGAN.CT2', Workload(env.BATCH,  8,  8,  512, 256, 4, 4, 1, 1, 2, 2)),
    ('DCGAN.CT3', Workload(env.BATCH, 16, 16,  256, 128, 4, 4, 1, 1, 2, 2)),
]

# FIXME: we need a custom clip operator to circumvent a pattern detection limitation
@tvm.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    x = tvm.compute(x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
    x = tvm.compute(x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return x

# Helper function to get factors
def _find_factors(n):
    factors = []
    for f in range(1, n + 1):
        if n % f == 0:
            factors.append(f)
    return factors


def run_conv2d_transpose(env, remote, wl, target,
               check_correctness=True, print_ir=False,
               samples=4):

    # Workload assertions
    assert wl.hpad == wl.wpad

    # Perform packing only if we are targeting the accelerator
    if "arm_cpu" in target.keys:
        data_pack = False
        layout = "NCHW"
    elif "vta" in target.keys:
        data_pack = True
        layout = "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN)

    # Derive shapes depending upon packing

    a_shape = (wl.batch, wl.in_filter, wl.height, wl.width)
    w_shape = (wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)
    b_shape = (wl.batch, wl.out_filter, 1, 1)
    if data_pack:
        data_shape = (wl.batch//env.BATCH, wl.in_filter//env.BLOCK_IN,
                      wl.height, wl.width, env.BATCH, env.BLOCK_IN)
        kernel_shape = (wl.out_filter//env.BLOCK_OUT, wl.in_filter//env.BLOCK_IN,
                        wl.hkernel, wl.wkernel, env.BLOCK_OUT, env.BLOCK_IN)
        bias_shape = (wl.batch//env.BATCH, wl.out_filter//env.BLOCK_OUT,
                      1, 1, env.BATCH, env.BLOCK_OUT)
    else:
        data_shape = a_shape
        kernel_shape = w_shape
        bias_shape = b_shape
    data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = tvm.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
    bias = tvm.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)

    # Define base computation schedule
    with target:
        res = topi.nn.conv2d_transpose_nchw(
            data, kernel, (wl.hstride, wl.wstride), (wl.hpad, wl.wpad), env.acc_dtype)
        res = topi.right_shift(res, 8)
        res = topi.add(res, bias)
        res = my_clip(res, 0, (1 << env.OUT_WIDTH - 1) - 1)
        res = topi.cast(res, env.out_dtype)
        # Derive base schedule
        s = topi.generic.schedule_conv2d_transpose_nchw([res])
        if print_ir:
            print(vta.lower(s, [data, kernel, bias, res], simple_mode=True))

    # Derive number of ops
    fout_height = (wl.height + 2 * wl.hpad - wl.hkernel) // wl.hstride + 1
    fout_width = (wl.width + 2 * wl.wpad - wl.wkernel) // wl.wstride + 1
    num_ops = 2 * wl.batch * fout_height * fout_width * wl.hkernel * wl.wkernel * wl.out_filter * wl.in_filter

    # @memoize("vta.tests.test_benchmark_topi.conv2d.verify_nhwc")
    def get_ref_data():
        # derive min max for act, wgt, and bias types (max non inclusive)
        a_min, a_max = 0 - (1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
        w_min, w_max = 0 - (1 << (env.WGT_WIDTH - 1)), (1 << (env.WGT_WIDTH - 1))
        b_min, b_max = 0 - 1 << (env.INP_WIDTH + env.WGT_WIDTH - 2), 1 << (env.INP_WIDTH + env.WGT_WIDTH - 2)
        a_np = np.random.randint(a_min, a_max, size=a_shape).astype(data.dtype)
        w_np = np.random.randint(w_min, w_max, size=(wl.in_filter, wl.out_filter, wl.hkernel, wl.wkernel)).astype(kernel.dtype)
        b_np = np.random.randint(b_min, b_max, size=b_shape).astype(env.acc_dtype)
        r_np = topi.testing.conv2d_transpose_nchw_python(
            a_np.astype(env.acc_dtype), w_np.astype(env.acc_dtype), (wl.hstride, wl.wstride), wl.hpad).astype(env.acc_dtype)
        return a_np, w_np, b_np, r_np

    # Data in original format
    data_np, kernel_np, bias_np, res_ref = get_ref_data()
    if data_pack:
        data_np = data_np.reshape(
            wl.batch//env.BATCH, env.BATCH,
            wl.in_filter//env.BLOCK_IN, env.BLOCK_IN,
            wl.height, wl.width).transpose((0, 2, 4, 5, 1, 3))
        kernel_np = kernel_np.reshape(
            wl.out_filter//env.BLOCK_OUT, env.BLOCK_OUT,
            wl.in_filter//env.BLOCK_IN, env.BLOCK_IN,
            wl.hkernel, wl.wkernel).transpose((0, 2, 4, 5, 1, 3))
        bias_np = bias_np.reshape(
            wl.batch//env.BATCH, wl.out_filter//env.BLOCK_OUT,
            1, 1, env.BATCH, env.BLOCK_OUT)

    # Build
    if "vta" in target.keys:
        mod = vta.build(s, [data, kernel, bias, res],
                        target=target,
                        target_host=env.target_host,
                        name="conv2d_transpose")
    else:
        mod = tvm.build(s, [data, kernel, bias, res],
                        target=target,
                        target_host=env.target_host,
                        name="conv2d_transpose")
    temp = util.tempdir()
    mod.save(temp.relpath("conv2d_transpose.o"))
    remote.upload(temp.relpath("conv2d_transpose.o"))
    f = remote.load_module("conv2d_transpose.o")
    ctx = remote.context(str(target))

    res_np = np.zeros(topi.util.get_const_tuple(res.shape)).astype(res.dtype)
    data_arr = tvm.nd.array(data_np, ctx)
    kernel_arr = tvm.nd.array(kernel_np, ctx)
    bias_arr = tvm.nd.array(bias_np, ctx)
    res_arr = tvm.nd.array(res_np, ctx)
    time_f = f.time_evaluator("conv2d_transpose", ctx, number=samples)

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
            cost = time_f(data_arr, kernel_arr, bias_arr, res_arr)
            if env.TARGET == "sim":
                stats = json.loads(remote.get_function("vta.simulator.profiler_status")())
            else:
                stats = json.loads(remote.get_function("vta.tsim.profiler_status")())
        else:
            simulator.clear_stats()
            cost = time_f(data_arr, kernel_arr, bias_arr, res_arr)
            stats = simulator.stats()
    else:
        cost = time_f(data_arr, kernel_arr, bias_arr, res_arr)

    # Check correctness
    correct = False
    if check_correctness:
        res_orig = res_arr.asnumpy()
        if data_pack:
            res_orig = res_orig.transpose(
                (0, 4, 1, 5, 2, 3)).reshape(wl.batch, wl.out_filter, fout_height, fout_width)
            bias_np = bias_np.transpose(
                (0, 4, 1, 5, 2, 3)).reshape(wl.batch, wl.out_filter, 1, 1)
        res_ref = res_ref >> 8
        res_ref += bias_np
        res_ref = np.clip(res_ref, 0, (1 << env.OUT_WIDTH - 1) - 1)
        res_ref = res_ref.astype(env.out_dtype)
        correct = np.allclose(res_orig, res_ref)

    gops = (num_ops / cost.mean) / float(10 ** 9)
    status = "PASSED" if correct else "FAILED"
    if "arm_cpu" in target.keys:
        device = "CPU"
    elif "vta" in target.keys:
        device = "VTA"
    print("%s CONV2D TEST %s: Time cost = %g sec/op, %g GOPS" % (device, status, cost.mean, gops))

    return correct, cost, stats

def test_conv2d_transpose(device="vta"):
    def _run(env, remote):
        if device == "vta":
            target = env.target
            if env.TARGET not in ["sim", "tsim"]:
                assert tvm.module.enabled("rpc")
                program_fpga(remote, bitstream=None)
                reconfig_runtime(remote)
        elif device == "arm_cpu":
            target = env.target_vta_cpu
        with autotvm.tophub.context(target): # load pre-tuned schedule parameters
            for _, wl in dcgan_wklds:
                print(wl)
                run_conv2d_transpose(env, remote, wl, target)
    vta.testing.run(_run)

if __name__ == "__main__":
    # test_conv2d_transpose(device="arm_cpu")
    test_conv2d_transpose(device="vta")


# def test_vta_conv2d_transpose():
#     def run_vta_conv2d_transpose(env, remote, name, wl, profile=True):
#         assert wl.batch % env.BATCH == 0
#         assert wl.in_filter % env.BLOCK_IN == 0
#         assert wl.out_filter % env.BLOCK_OUT == 0

#         data_shape = (wl.batch//env.BATCH, wl.in_filter//env.BLOCK_IN,
#                       wl.height, wl.width, env.BATCH, env.BLOCK_IN)
#         kernel_shape = (wl.out_filter//env.BLOCK_OUT, wl.in_filter // env.BLOCK_IN,
#                         wl.hkernel, wl.wkernel, env.BLOCK_OUT, env.BLOCK_IN)
#         bias_shape = (wl.batch//env.BATCH, wl.out_filter//env.BLOCK_OUT,
#                       1, 1, env.BATCH, env.BLOCK_OUT)

#         fout_height = (wl.height - 1) * wl.hstride - 2 * wl.hpad + wl.hkernel
#         fout_width = (wl.width - 1) * wl.wstride - 2 * wl.wpad + wl.wkernel

#         data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
#         kernel = tvm.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
#         bias = tvm.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)

#         res_conv = vta.top.packed_conv2d_transpose(
#             data, kernel, (wl.hpad, wl.wpad), (wl.hstride, wl.wstride))
#         res = topi.right_shift(res_conv, 8)
#         res = topi.add(res, bias)
#         res = my_clip(res, 0, 127)
#         res = topi.cast(res, "int8")

#         # To compute number of ops, use a x2 factor for FMA
#         num_ops = 2 * wl.batch * fout_height * fout_width * wl.hkernel * wl.wkernel * \
#                   wl.out_filter * wl.in_filter / (wl.hstride * wl.wstride)

#         a_shape = (wl.batch, wl.in_filter, wl.height, wl.width)
#         w_shape = (wl.in_filter, wl.out_filter, wl.hkernel, wl.wkernel)
#         data_dtype = data.dtype
#         kernel_dtype = kernel.dtype
#         acc_dtype = env.acc_dtype
#         stride = (wl.hstride, wl.wstride)
#         padding = (wl.hpad, wl.wpad)

#         # @memoize("vta.tests.test_conv2d_transpose")
#         def get_ref_data():
#             a_np = (np.random.uniform(size=a_shape) * 4).astype(data_dtype)
#             w_np = (np.random.uniform(size=w_shape) * 4).astype(kernel_dtype)
#             a_np = np.abs(a_np)
#             w_np = np.abs(w_np)
#             b_np = topi.testing.conv2d_transpose_nchw_python(
#                 a_np.astype(acc_dtype), w_np.astype(acc_dtype), stride, padding).astype(acc_dtype)
#             return a_np, w_np, b_np

#         def verify(s, check_correctness):
#             mod = vta.build(s, [data, kernel, bias, res], "ext_dev",
#                             env.target_host, name="conv2d_transpose")
#             temp = util.tempdir()

#             mod.save(temp.relpath("conv2d_transpose.o"))
#             remote.upload(temp.relpath("conv2d_transpose.o"))
#             f = remote.load_module("conv2d_transpose.o")
#             # verify
#             ctx = remote.ext_dev(0)
#             # Data in original format
#             data_orig, kernel_orig, res_ref = get_ref_data()
#             bias_orig = (np.random.uniform(size=(wl.out_filter,)) * 4).astype("int32")
#             bias_orig = np.abs(bias_orig)

#             data_packed = data_orig.reshape(
#                 wl.batch//env.BATCH, env.BATCH,
#                 wl.in_filter//env.BLOCK_IN, env.BLOCK_IN,
#                 wl.height, wl.width).transpose((0, 2, 4, 5, 1, 3))
#             kernel_packed = kernel_orig.reshape(
#                 wl.in_filter//env.BLOCK_IN, env.BLOCK_IN,
#                 wl.out_filter//env.BLOCK_OUT, env.BLOCK_OUT,
#                 wl.hkernel, wl.wkernel).transpose((2, 0, 4, 5, 3, 1))
#             kernel_flipped = np.flip(kernel_packed, [2, 3])

#             bias_packed = bias_orig.reshape(
#                 1, wl.out_filter // env.BLOCK_OUT, 1, 1, env.BATCH, env.BLOCK_OUT)
#             res_shape = topi.util.get_const_tuple(res.shape)

#             res_np = np.zeros(res_shape).astype(res.dtype)
#             data_arr = tvm.nd.array(data_packed, ctx)
#             kernel_arr = tvm.nd.array(kernel_flipped, ctx)
#             bias_arr = tvm.nd.array(bias_packed, ctx)
#             res_arr = tvm.nd.array(res_np, ctx)

#             remote.get_function("vta.simulator.profiler_clear")()
#             time_f = f.time_evaluator("conv2d_transpose", ctx, number=1)
#             cost = time_f(data_arr, kernel_arr, bias_arr, res_arr)
#             stats = json.loads(remote.get_function("vta.simulator.profiler_status")())

#             res_unpack = res_arr.asnumpy().transpose(
#                 (0, 4, 1, 5, 2, 3)).reshape(wl.batch, wl.out_filter, fout_height, fout_width)
#             if check_correctness:
#                 assert wl.hpad == wl.wpad
#                 stride = (wl.hstride, wl.wstride)
#                 padding = (wl.hpad, wl.wpad)
#                 res_ref = res_ref >> 8
#                 res_ref += bias_orig.reshape(wl.out_filter, 1, 1)
#                 res_ref = np.clip(res_ref, 0, 127).astype("int8")
#                 np.testing.assert_allclose(res_unpack, res_ref)
#             return cost, stats

#         def conv2d_transpose_normal(print_ir):
#             # print("----- Conv2d Transpose End-to-End Test-------")
#             with vta.build_config():
#                 s = vta.top.schedule_packed_conv2d_transpose([res])
#                 if print_ir:
#                     print(vta.lower(s, [data, kernel, bias, res], simple_mode=True))
#             cost, stats = verify(s, True)
#             # gops = (num_ops / cost.mean) / float(10 ** 9)
#             # print("\tTime cost = %g sec/op, %g GOPS" % (cost.mean, gops))
#             return cost, stats

#         return conv2d_transpose_normal(False)

#     def _run(env, remote):
        

#         for tsk in tasks:
#             print(tsk)
#             name, wkl = tsk
#             run_vta_conv2d_transpose(env, remote, name, wkl)
#         return

#         # TUNER
#         map_list = {}
#         for i, tsk in enumerate(tasks):
#             print(tsk)
#             name, wkl = tsk

#             fout_height = (wkl.height - 1) * wkl.hstride - 2 * wkl.hpad + wkl.hkernel
#             fout_width = (wkl.width - 1) * wkl.wstride - 2 * wkl.wpad + wkl.wkernel

#             batch_factors = _find_factors(wkl.batch // env.BATCH)
#             height_factors = _find_factors(fout_height)
#             width_factors = _find_factors(fout_width)
#             cin_factors = _find_factors(wkl.in_filter // env.BLOCK_IN)
#             cout_factors = _find_factors(wkl.out_filter // env.BLOCK_OUT)
#             ht_factors = [1]
#             cot_factors = [1]

#             sch_list = []
#             cost_list = []
#             ct = 0
#             total = np.prod([len(x) for x in [batch_factors, height_factors, width_factors, cin_factors, cout_factors,
#                                               ht_factors, cot_factors]])
#             best = 1 << 32
#             for b_f in batch_factors:
#                 for h_f in height_factors:
#                     for w_f in width_factors:
#                         for ci_f in cin_factors:
#                             for co_f in cout_factors:
#                                 for h_t in ht_factors:
#                                     for co_t in cot_factors:
#                                         sch = Schedule(b_f, co_f, ci_f, h_f, w_f, h_t, co_t, False)
#                                         inject_schedule(sch)
#                                         try:
#                                             _, stats = run_vta_conv2d_transpose(env, remote, name, wkl)
#                                             cost = stats['inp_load_nbytes'] + stats['wgt_load_nbytes'] + stats['acc_load_nbytes'] + \
#                                                    stats['out_store_nbytes'] + stats['uop_load_nbytes']
#                                         except tvm.TVMError:
#                                             cost = 1 << 32
#                                         best = min(best, cost)
#                                         print("[Task %d/%d] %d/%d : %d / %d" % (i, len(tasks), ct, total, cost, best))
#                                         ct += 1
#                                         sch_list.append(sch)
#                                         cost_list.append(cost)
#             cost_list = np.array(cost_list)

#             sort_index = np.argsort(cost_list)

#             map_list[str(wkl)] = tuple(sch_list[sort_index[0]])

#         # pickle.dump(map_list, open("conv_tmp.pkl", "wb"))

#     vta.testing.run(_run)

# if __name__ == "__main__":
#     test_vta_conv2d_transpose()
