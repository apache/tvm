import os
import tvm
import mxnet as mx
import vta
import numpy as np
import topi
from collections import namedtuple
from tvm.contrib import rpc, util
import pandas as pd

host = "pynq"
port = 9091
target = "llvm -target=armv7-none-linux-gnueabihf -mattr=+neon"
out_dtype = "int%d" % vta.VTA_OUT_WIDTH
inp_dtype = "int%d" % vta.VTA_INP_WIDTH
wgt_dtype = "int%d" % vta.VTA_WGT_WIDTH

Workload = namedtuple("Conv2DWorkload",
                      ['batch', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

class Conv2DSchedule(object):
    def __init__(self,
                 b_factor=1,
                 oc_factor=1,
                 ko_factor=1,
                 h_factor=1,
                 w_factor=0,
                 oc_nthread=0,
                 h_nthread=0,
                 debug_sync=False):
        self.b_factor = b_factor
        self.oc_factor = oc_factor
        self.ko_factor = ko_factor
        self.h_factor = h_factor
        self.w_factor = w_factor
        self.oc_nthread = oc_nthread
        self.h_nthread = h_nthread
        self.debug_sync = debug_sync
    def __str__(self):
        return "{}.{}.{}.{}.{}.{}.{}".format(
            self.b_factor, self.oc_factor, self.ko_factor,
            self.h_factor, self.w_factor,
            self.oc_nthread, self.h_nthread)

Schedule = Conv2DSchedule

def get_insn_count(layer, sched):
    b, h, w, ci, co = sched
    b_factor = b
    h_factor = layer.height / h
    w_factor = layer.width / w
    ci_factor = int(np.ceil(float(layer.in_filter) / (ci * vta.VTA_BLOCK_IN)))
    co_factor = int(np.ceil(float(layer.out_filter) / (co * vta.VTA_BLOCK_OUT)))
    input_xfers = b_factor * h_factor * w_factor * co_factor * ci_factor
    weight_xfers = b_factor * h_factor * w_factor * co_factor * ci_factor
    output_xfers = b_factor * h_factor * w_factor * co_factor
    # compute instruction count
    # output transfer factor: 4 (INIT, GEMM, ALU, STORE)
    # offset: 5 (3 uop kernels, 1 initial dep push, 1 finish, co_factor)
    insn_count = input_xfers + weight_xfers + (output_xfers * 4) + 5 + co_factor
    return insn_count

def find_schedules(layer, mtOnly=False, bestOnly=False):
    # Helper function to get factors
    def find_factors(n):
        factors = []
        for i in range(1, n+1):
            if n % i == 0:
                factors.append(i)
        return factors
    # Scheduling exploration
    batch_factors = find_factors(int(np.ceil(float(layer.batch) / vta.VTA_BATCH)))
    height_factors = find_factors(layer.height / layer.hstride)
    width_factors = find_factors(layer.width / layer.wstride)
    cin_factors = find_factors(int(np.ceil(float(layer.in_filter) / vta.VTA_BLOCK_IN)))
    cout_factors = find_factors(int(np.ceil(float(layer.out_filter) / vta.VTA_BLOCK_OUT)))
    ht_factors = [1, 2]
    cot_factors = [1, 2]
    # Explore schedules
    schedules = []
    for b in batch_factors:
        for h in height_factors:
            for w in width_factors:
                for ci in cin_factors:
                    for co in cout_factors:
                        # FIXME: filter because of 2D load
                        if w == layer.width/layer.wstride or (w != layer.width/layer.wstride and co == 1):
                            # FIXME: filter because of 2D load
                            if ci == 1:
                                schedules.append([b, h, w, ci, co])
    # Filter the schedules that wouldn't work in the available BRAM sizes
    input_elem_size_b = vta.VTA_BATCH * vta.VTA_BLOCK_IN * vta.VTA_INP_WIDTH
    weight_elem_size_b = vta.VTA_BLOCK_IN * vta.VTA_BLOCK_OUT * vta.VTA_WGT_WIDTH
    output_elem_size_b = vta.VTA_BATCH * vta.VTA_BLOCK_OUT * vta.VTA_OUT_WIDTH
    input_brams_capacity_b = vta.VTA_INP_BUFF_SIZE * 8
    weight_brams_capacity_b = vta.VTA_WGT_BUFF_SIZE * 8
    output_brams_capacity_b = vta.VTA_OUT_BUFF_SIZE * 8
    fil_sched = []
    xfer_size = []
    for sched in schedules:
        b, h, w, ci, co = sched
        for ht in [1, 2]:
            for cot in [1, 2]:
                # Make sure to filter cases where we apply threading on two axes
                # or cases where the threading factors for h and co are not
                # factors of h and co
                if not (ht == 2 and cot == 2) and h % ht == 0 and co % cot == 0:
                    # If in multi-threaded mode, only allow for mt configs:
                    if (mtOnly and (ht == 2 or cot == 2)) or not mtOnly:
                        h /= ht
                        co /= cot
                        input_tile_elems = b * \
                                ((h - 1) * layer.hstride + layer.hkernel) * \
                                ((w - 1) * layer.wstride + layer.wkernel) * ci
                        weight_tile_elems = layer.hkernel * layer.wkernel * ci * co
                        output_tile_elems = b * h * w * co
                        insn_count = get_insn_count(layer, sched)
                        # 1. Check input capacity
                        # 2. Check weight capacity
                        # 3. Check output capacity
                        # 4. Check instruction capacity
                        # 5. Make sure that we don't write to the same acc location
                        #    within 2 consecutive cycles
                        if input_tile_elems*input_elem_size_b <= input_brams_capacity_b/(cot*ht) and \
                           weight_tile_elems*weight_elem_size_b <= weight_brams_capacity_b and \
                           output_tile_elems*output_elem_size_b <= output_brams_capacity_b/(cot*ht) and \
                           insn_count <= vta.VTA_MAX_XFER / 16 and \
                           h > 2 and w > 2:
                            schedule = Schedule(oc_factor=co, ko_factor=ci, h_factor=h,
                                                w_factor=w, oc_nthread=cot, h_nthread=ht)
                            fil_sched.append(schedule)
                            xfer_size.append(get_data_movementB(schedule, layer))
    if bestOnly:
        return [fil_sched[xfer_size.index(min(xfer_size))]]
    else:
        return fil_sched

def get_data_movementB(sched, layer):
    # Derive data movement
    input_elem_size_b = vta.VTA_BATCH * vta.VTA_BLOCK_IN * vta.VTA_INP_WIDTH
    weight_elem_size_b = vta.VTA_BLOCK_IN * vta.VTA_BLOCK_OUT * vta.VTA_WGT_WIDTH
    output_elem_size_b = vta.VTA_BATCH * vta.VTA_BLOCK_OUT * vta.VTA_OUT_WIDTH
    b = sched.b_factor
    h = sched.h_factor
    w = sched.w_factor
    ci = sched.ko_factor
    co = sched.oc_factor
    ht = sched.h_nthread
    cot = sched.oc_nthread
    input_tile_elems = b * \
            ((h - 1) * layer.hstride + layer.hkernel) * \
            ((w - 1) * layer.wstride + layer.wkernel) * ci
    weight_tile_elems = layer.hkernel * layer.wkernel * ci
    output_tile_elems = b * h * w * co
    # Derive factors
    b_factor = int(np.ceil(float(layer.batch) / (b * vta.VTA_BATCH)))
    h_factor = (layer.height / layer.hstride) / h
    w_factor = (layer.width / layer.wstride) / w
    ci_factor = int(np.ceil(float(layer.in_filter) / (ci * vta.VTA_BLOCK_IN)))
    co_factor = int(np.ceil(float(layer.out_filter) / (co * vta.VTA_BLOCK_OUT)))
    # Derive transfers
    input_xfers = b_factor * h_factor * w_factor * co_factor * ci_factor
    weight_xfers = b_factor * h_factor * w_factor * co_factor * ci_factor
    output_xfers = b_factor * h_factor * w_factor * co_factor
    # Compute total transfer sizes
    input_xfer_B = input_tile_elems * input_xfers * input_elem_size_b / 8
    weight_xfer_B = weight_tile_elems * weight_xfers * weight_elem_size_b / 8
    output_xfer_B = output_tile_elems * output_xfers * output_elem_size_b / 8
    total_xfer_B = input_xfer_B + weight_xfer_B + output_xfer_B
    return total_xfer_B

def test_conv2d_chwv(layer, key, batch_size, wl, sched, log_frame, profile=True):
    assert batch_size % vta.VTA_BATCH == 0
    assert wl.in_filter % vta.VTA_BLOCK_IN == 0
    assert wl.out_filter % vta.VTA_BLOCK_OUT == 0
    data_shape = (batch_size//vta.VTA_BATCH, wl.in_filter//vta.VTA_BLOCK_IN,
                  wl.height, wl.width, vta.VTA_BATCH, vta.VTA_BLOCK_IN)
    kernel_shape = (wl.out_filter//vta.VTA_BLOCK_OUT, wl.in_filter//vta.VTA_BLOCK_IN,
                    wl.hkernel, wl.wkernel, vta.VTA_BLOCK_OUT, vta.VTA_BLOCK_IN)
    fout_height = (wl.height + 2 * wl.hpad - wl.hkernel) // wl.hstride + 1
    fout_width = (wl.width + 2 * wl.wpad - wl.wkernel) // wl.wstride + 1
    res_shape = (batch_size//vta.VTA_BATCH, wl.out_filter//vta.VTA_BLOCK_OUT,
                 fout_height, fout_width, vta.VTA_BATCH, vta.VTA_BLOCK_OUT)
    data = tvm.placeholder(data_shape, name="data", dtype=inp_dtype)
    kernel = tvm.placeholder(kernel_shape, name="kernel", dtype=wgt_dtype)
    if wl.hpad or wl.wpad:
        data_buf = topi.nn.pad(data, [0, 0, wl.hpad, wl.wpad, 0, 0], name="data_buf")
    else:
        data_buf = tvm.compute(data_shape, lambda *i: data(*i), "data_buf")
    kernel_buf = tvm.compute(kernel_shape, lambda *i: kernel(*i), "kernel_buf")
    di = tvm.reduce_axis((0, wl.hkernel), name='di')
    dj = tvm.reduce_axis((0, wl.wkernel), name='dj')
    ko = tvm.reduce_axis((0, wl.in_filter//vta.VTA_BLOCK_IN), name='ko')
    ki = tvm.reduce_axis((0, vta.VTA_BLOCK_IN), name='ki')
    res_cnv = tvm.compute(
        res_shape,
        lambda bo, co, i, j, bi, ci: tvm.sum(
            data_buf[bo, ko, i*wl.hstride+di, j*wl.wstride+dj, bi, ki].astype(out_dtype) *
            kernel_buf[co, ko, di, dj, ci, ki].astype(out_dtype),
        axis=[ko, di, dj, ki]),
        name="res_cnv")
    res_shf = tvm.compute(res_shape, lambda *i: res_cnv(*i) >> 8, name="res_shf")
    res = tvm.compute(res_shape, lambda *i: res_shf(*i).astype(inp_dtype), name="res")
    num_ops = batch_size * fout_height * fout_width * wl.hkernel * wl.wkernel * wl.out_filter * wl.in_filter
    total_xfer_B = get_data_movementB(sched, wl)

    def verify(s, check_correctness):
        mod = tvm.build(s, [data, kernel, res], "ext_dev", target, name="conv2d")
        temp = util.tempdir()
        remote = rpc.connect(host, port)
        mod.save(temp.relpath("conv2d.o"))
        remote.upload(temp.relpath("conv2d.o"))
        f = remote.load_module("conv2d.o")
        # verify
        ctx = remote.ext_dev(0)
        # Data in original format
        data_orig = np.random.randint(
            -128, 128, size=(batch_size, wl.in_filter, wl.height, wl.width)).astype(data.dtype)
        kernel_orig = np.random.randint(
            -128, 128, size=(wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)).astype(kernel.dtype)
        data_packed = data_orig.reshape(
            batch_size//vta.VTA_BATCH, vta.VTA_BATCH,
            wl.in_filter//vta.VTA_BLOCK_IN, vta.VTA_BLOCK_IN,
            wl.height, wl.width).transpose((0, 2, 4, 5, 1, 3))
        kernel_packed = kernel_orig.reshape(
            wl.out_filter//vta.VTA_BLOCK_OUT, vta.VTA_BLOCK_OUT,
            wl.in_filter//vta.VTA_BLOCK_IN, vta.VTA_BLOCK_IN,
            wl.hkernel, wl.wkernel).transpose((0, 2, 4, 5, 1, 3))
        res_np = np.zeros(res_shape).astype(res.dtype)
        data_arr = tvm.nd.array(data_packed, ctx)
        kernel_arr = tvm.nd.array(kernel_packed, ctx)
        res_arr = tvm.nd.array(res_np, ctx)
        time_f = f.time_evaluator("conv2d", ctx, number=1)
        cost = time_f(data_arr, kernel_arr, res_arr)
        res_unpack = res_arr.asnumpy().transpose(
            (0, 4, 1, 5, 2, 3)).reshape(batch_size, wl.out_filter, fout_height, fout_width)
        if check_correctness:
            res_ref = mx.nd.Convolution(
                mx.nd.array(data_orig.astype(out_dtype), mx.cpu(0)),
                mx.nd.array(kernel_orig.astype(out_dtype), mx.cpu(0)),
                stride=(wl.hstride, wl.wstride),
                kernel=(wl.hkernel, wl.wkernel),
                num_filter=wl.out_filter,
                no_bias=True,
                pad=(wl.hpad, wl.wpad)).asnumpy().astype(out_dtype)
            res_ref = np.right_shift(res_ref, 8).astype(res.dtype)
            np.testing.assert_allclose(res_unpack, res_ref)
            print("Correctness check pass...")
        return cost

    def run_schedule(load_inp, load_wgt, gemm, alu, store_out,
                     print_ir, check_correctness):
        # schedule1
        s = tvm.create_schedule(res.op)
        s[data_buf].set_scope(vta.SCOPE_INP)
        s[kernel_buf].set_scope(vta.SCOPE_WGT)
        s[res_cnv].set_scope(vta.SCOPE_OUT)
        s[res_shf].set_scope(vta.SCOPE_OUT)
        # tile
        oc_factor = (sched.oc_factor if sched.oc_factor
                     else wl.out_filter // vta.VTA_BLOCK_OUT)
        h_factor = (sched.h_factor if sched.h_factor else fout_height)
        w_factor = (sched.w_factor if sched.w_factor else fout_width)
        xbo, xco, xi, xj, xbi, xci = s[res].op.axis
        xco0, xco1 = s[res].split(xco, factor=oc_factor)
        xi0, xi1 = s[res].split(xi, factor=h_factor)
        xj0, xj1 = s[res].split(xj, factor=w_factor)
        s[res].reorder(xbo, xi0, xco0, xj0, xco1, xi1, xj1, xbi, xci)
        s[res_cnv].compute_at(s[res], xj0)
        s[res_shf].compute_at(s[res], xj0)

        if sched.oc_nthread > 1:
            _, tx = s[res].split(xco0, factor=sched.oc_nthread)
            s[res].reorder(tx, xbo)
            s[res].bind(tx, tvm.thread_axis("cthread"))

        if sched.h_nthread > 1:
            xo, tx = s[res].split(xi0, factor=sched.h_nthread)
            s[res].reorder(tx, xbo)
            s[res].bind(tx, tvm.thread_axis("cthread"))

        xbo, xco, xi, xj, xbi, xci = s[res_cnv].op.axis
        s[res_cnv].reorder(xbo, ko, xj, dj, di, xco, xi, xbi, xci, ki)

        if sched.ko_factor:
            ko0, ko1 = s[res_cnv].split(ko, factor=sched.ko_factor)
            s[data_buf].compute_at(s[res_cnv], ko0)
            s[kernel_buf].compute_at(s[res_cnv], ko0)
        # Use VTA instructions
        s[data_buf].pragma(s[data_buf].op.axis[0], load_inp)
        s[kernel_buf].pragma(s[kernel_buf].op.axis[0], load_wgt)
        s[res_cnv].tensorize(xbi, gemm)
        s[res_shf].pragma(s[res_shf].op.axis[0], alu)
        s[res].pragma(xco1, store_out)
        if sched.debug_sync:
            s[res].pragma(xco0, "coproc_sync")
        if print_ir:
            print(tvm.lower(s, [data, kernel, res], simple_mode=True))
        return verify(s, check_correctness)

    def conv_normal(print_ir):
        print("----- CONV2D End-to-End Test-------")
        def run_test(header, print_ir, check_correctness):
            s = [1, sched.oc_factor, sched.ko_factor, sched.h_factor, sched.w_factor]
            cost = run_schedule(
                vta.DMA_COPY, vta.DMA_COPY,
                vta.GEMM, vta.ALU, vta.DMA_COPY,
                print_ir, check_correctness)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS" % (cost.mean, gops))
            log_frame["key"].append(key)
            log_frame["layer"].append(layer)
            log_frame["total-data"].append(total_xfer_B)
            log_frame["total-gops"].append(gops)
            log_frame["total-cost"].append(cost.mean)
            log_frame["total-insn"].append(get_insn_count(wl, s))
            log_frame["block-batch"].append(vta.VTA_BATCH)
            log_frame["block-in"].append(vta.VTA_BLOCK_IN)
            log_frame["block-out"].append(vta.VTA_BLOCK_OUT)
            log_frame["inp-width"].append(vta.VTA_INP_WIDTH)
            log_frame["wgt-width"].append(vta.VTA_WGT_WIDTH)
            log_frame["uop-size"].append(vta.VTA_UOP_BUFF_SIZE)
            log_frame["inp-size"].append(vta.VTA_INP_BUFF_SIZE)
            log_frame["wgt-size"].append(vta.VTA_WGT_BUFF_SIZE)
            log_frame["out-size"].append(vta.VTA_OUT_BUFF_SIZE)
            log_frame["oc-factor"].append(sched.oc_factor)
            log_frame["ic-factor"].append(sched.ko_factor)
            log_frame["h-factor"].append(sched.h_factor)
            log_frame["w-factor"].append(sched.w_factor)
            log_frame["oc-threads"].append(sched.oc_nthread)
            log_frame["h-threads"].append(sched.h_nthread)
            log_frame["threaded"].append(True if sched.oc_nthread > 1 or sched.h_nthread > 1 else False)

        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir, True)

    def skip_alu_unittest(print_ir):
        mock = vta.mock
        print("----- Skip ALU Unit Test-------")
        def run_test(header, print_ir):
            cost = run_schedule(
                vta.DMA_COPY, vta.DMA_COPY,
                vta.GEMM, mock.ALU, vta.DMA_COPY,
                print_ir, False)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS" % (cost.mean, gops))
            log_frame["skip-alu-gops"].append(gops)
            log_frame["skip-alu-cost"].append(cost.mean)

        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir)
        print("")

    def gemm_unittest(print_ir):
        mock = vta.mock
        print("----- GEMM Unit Test-------")
        def run_test(header, print_ir):
            cost = run_schedule(
                mock.DMA_COPY, mock.DMA_COPY,
                vta.GEMM, mock.ALU, mock.DMA_COPY,
                print_ir, False)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS" % (cost.mean, gops))
            log_frame["gemm-gops"].append(gops)
            log_frame["gemm-cost"].append(cost.mean)
        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir)
        print("")

    def alu_unittest(print_ir):
        mock = vta.mock
        print("----- ALU Unit Test-------")
        def run_test(header, print_ir):
            cost = run_schedule(
                mock.DMA_COPY, mock.DMA_COPY,
                mock.GEMM, vta.ALU, mock.DMA_COPY,
                print_ir, False)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS" % (cost.mean, gops))
            log_frame["alu-gops"].append(gops)
            log_frame["alu-cost"].append(cost.mean)
        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir)
        print("")

    def load_inp_unittest(print_ir):
        mock = vta.mock
        print("----- LoadInp Unit Test-------")
        def run_test(header, print_ir):
            cost = run_schedule(
                vta.DMA_COPY, mock.DMA_COPY,
                mock.GEMM, mock.ALU, mock.DMA_COPY,
                print_ir, False)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            bandwith = (batch_size * wl.in_filter * wl.height *
                        wl.width * vta.INP_WIDTH / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS, bandwith=%g gbits" % (
                cost.mean, gops, bandwith))
            log_frame["ld-inp-gbits"].append(bandwith)
            log_frame["ld-inp-cost"].append(cost.mean)
        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir)
        print("")

    def load_wgt_unittest(print_ir):
        mock = vta.mock
        print("----- LoadWgt Unit Test-------")
        def run_test(header, print_ir):
            cost = run_schedule(
                mock.DMA_COPY, vta.DMA_COPY,
                mock.GEMM, mock.ALU, mock.DMA_COPY, print_ir,
                False)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            bandwith = (wl.out_filter * wl.in_filter * wl.hkernel *
                        wl.wkernel * vta.WGT_WIDTH / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS, bandwith=%g gbits" % (
                cost.mean, gops, bandwith))
            log_frame["ld-wgt-gbits"].append(bandwith)
            log_frame["ld-wgt-cost"].append(cost.mean)
        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir)
        print("")

    def store_out_unittest(print_ir):
        mock = vta.mock
        print("----- StoreOut Unit Test-------")
        def run_test(header, print_ir):
            cost = run_schedule(
                mock.DMA_COPY, mock.DMA_COPY,
                mock.GEMM, mock.ALU, vta.DMA_COPY, print_ir,
                False)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            bandwith = (batch_size * wl.out_filter * fout_height *
                        fout_width * vta.OUT_WIDTH / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS, bandwith=%g gbits" % (
                cost.mean, gops, bandwith))
            log_frame["st-out-gbits"].append(bandwith)
            log_frame["st-out-cost"].append(cost.mean)
        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir)
        print("")

    def manual_unittest(print_ir):
        # Manual section used to teak the components
        mock = vta.mock
        print("----- Manual Unit Test-------")
        def run_test(header, print_ir):
            cost = run_schedule(
                vta.DMA_COPY, vta.DMA_COPY,
                vta.GEMM, vta.ALU, mock.DMA_COPY, print_ir,
                False)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS" % (
                cost.mean, gops))
        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir)
        print("")

    print("=================================")
    print("key=%s" % key)
    print(wl)
    conv_normal(False)
    if not profile:
        return
    skip_alu_unittest(False)
    gemm_unittest(False)
    alu_unittest(False)
    load_inp_unittest(False)
    load_wgt_unittest(False)
    store_out_unittest(False)

# Perform profiling
profile = False
# Data set batch size
batch_size = 1
# Use multi-threading for latency hiding
multi_threaded = False

# ResNet18 workloads
resnet = {
    # Workloads of resnet18 on imagenet
    0: Workload(1, 224, 224, 16, 64, 7, 7, 3, 3, 2, 2),
    1: Workload(1, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
    2: Workload(1, 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
    3: Workload(1, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
    4: Workload(1, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
    5: Workload(1, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
    6: Workload(1, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
    7: Workload(1, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
    8: Workload(1, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
    9: Workload(1, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
    10: Workload(1, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
    11: Workload(1, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
}

begin = 0
end = len(resnet)

resnet_schedules = []
for i in range(begin, end):
    scheds = find_schedules(resnet[i], mtOnly=multi_threaded, bestOnly=True)
    resnet_schedules.append([i, scheds])

keys = ["key", "layer", "total-data", "total-gops", "total-cost", "total-insn",
        "block-batch", "block-in", "block-out", "wgt-width", "inp-width",
        "uop-size", "inp-size", "wgt-size", "out-size",
        "oc-factor", "ic-factor", "h-factor", "w-factor",
        "oc-threads", "h-threads", "threaded"]

if profile:
    keys += ["skip-alu-gops", "skip-alu-cost",
             "gemm-gops", "gemm-cost", "alu-gops", "alu-cost",
             "ld-inp-cost", "ld-wgt-cost", "st-out-cost",
             "ld-inp-gbits", "ld-wgt-gbits", "st-out-gbits"]
log_frame = {
    k : [] for k in keys
}

for x in resnet_schedules:
    l, plans = x
    for plan in plans:
        key = "resnet-cfg[{}-{}]".format(l, plan)
        test_conv2d_chwv(l, key, batch_size, resnet[l], plan, log_frame, profile)

pd.set_option('expand_frame_repr', False)
log_df = pd.DataFrame()
for k  in keys:
    log_df[k] = log_frame[k]
print(log_df)

log_df.to_csv("conv2d.csv")
