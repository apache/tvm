import os
import tvm
import vta
import numpy as np
import time
from tvm.contrib import rpc, util

host = "pynq"
port = 9091
target = "llvm -target=armv7-none-linux-gnueabihf"
out_dtype = "int%d" % vta.VTA_OUT_WIDTH
inp_dtype = "int%d" % vta.VTA_INP_WIDTH
wgt_dtype = "int%d" % vta.VTA_WGT_WIDTH

def test_gemm_packed(batch_size, channel, block):
    data_shape = (batch_size//vta.VTA_BATCH,
                  channel//vta.VTA_BLOCK_IN,
                  vta.VTA_BATCH,
                  vta.VTA_BLOCK_IN)
    weight_shape = (channel//vta.VTA_BLOCK_OUT,
                    channel//vta.VTA_BLOCK_IN,
                    vta.VTA_BLOCK_OUT,
                    vta.VTA_BLOCK_IN)
    res_shape = (batch_size//vta.VTA_BATCH,
                 channel//vta.VTA_BLOCK_OUT,
                 vta.VTA_BATCH,
                 vta.VTA_BLOCK_OUT)
    num_ops = channel * channel * batch_size

    ko = tvm.reduce_axis((0, channel//vta.VTA_BLOCK_IN), name='ko')
    ki = tvm.reduce_axis((0, vta.VTA_BLOCK_IN), name='ki')

    data = tvm.placeholder(data_shape,
                           name="data",
                           dtype=inp_dtype)
    weight = tvm.placeholder(weight_shape,
                             name="weight",
                             dtype=wgt_dtype)
    data_buf = tvm.compute(data_shape,
                           lambda *i: data(*i),
                           "data_buf")
    weight_buf = tvm.compute(weight_shape,
                             lambda *i: weight(*i),
                             "weight_buf")
    res_gem = tvm.compute(res_shape,
                          lambda bo, co, bi, ci: tvm.sum(
                              data_buf[bo, ko, bi, ki].astype(out_dtype) *
                              weight_buf[co, ko, ci, ki].astype(out_dtype),
                              axis=[ko, ki]),
                          name="res_gem")
    res_shf = tvm.compute(res_shape,
                          lambda *i: res_gem(*i)>>8,
                          name="res_shf")
    res_max = tvm.compute(res_shape,
                          lambda *i: tvm.max(res_shf(*i), 0),
                          "res_max") #relu
    res_min = tvm.compute(res_shape,
                          lambda *i: tvm.min(res_max(*i), (1<<(vta.VTA_INP_WIDTH-1))-1),
                          "res_min") #relu
    res = tvm.compute(res_shape,
                      lambda *i: res_min(*i).astype(inp_dtype),
                      name="res")

    def verify(s, check_correctness=True):
        mod = tvm.build(s, [data, weight, res], "ext_dev", target, name="gemm")
        temp = util.tempdir()
        remote = rpc.connect(host, port)
        mod.save(temp.relpath("gemm.o"))
        remote.upload(temp.relpath("gemm.o"))
        f = remote.load_module("gemm.o")
        # verify
        ctx = remote.ext_dev(0)
        # Data in original format
        data_orig = np.random.randint(
            -128, 128, size=(batch_size, channel)).astype(data.dtype)
        weight_orig = np.random.randint(
            -128, 128, size=(channel, channel)).astype(weight.dtype)
        data_packed = data_orig.reshape(
            batch_size//vta.VTA_BATCH, vta.VTA_BATCH,
            channel//vta.VTA_BLOCK_IN, vta.VTA_BLOCK_IN).transpose((0, 2, 1, 3))
        weight_packed = weight_orig.reshape(
            channel//vta.VTA_BLOCK_OUT, vta.VTA_BLOCK_OUT,
            channel//vta.VTA_BLOCK_IN, vta.VTA_BLOCK_IN).transpose((0, 2, 1, 3))
        res_np = np.zeros(res_shape).astype(res.dtype)
        data_arr = tvm.nd.array(data_packed, ctx)
        weight_arr = tvm.nd.array(weight_packed, ctx)
        res_arr = tvm.nd.array(res_np, ctx)
        res_ref = np.zeros(res_shape).astype(out_dtype)
        for b in range(batch_size//vta.VTA_BATCH):
            for i in range(channel//vta.VTA_BLOCK_OUT):
                for j in range(channel//vta.VTA_BLOCK_IN):
                    res_ref[b,i,:] += np.dot(data_packed[b,j,:].astype(out_dtype),
                                             weight_packed[i,j].T.astype(out_dtype))
        res_ref = np.right_shift(res_ref, 8)
        res_ref = np.clip(res_ref, 0, (1<<(vta.VTA_INP_WIDTH-1))-1).astype(res.dtype)
        time_f = f.time_evaluator("gemm", ctx, number=20)
        cost = time_f(data_arr, weight_arr, res_arr)
        res_unpack = res_arr.asnumpy().reshape(batch_size//vta.VTA_BATCH,
                                               channel//vta.VTA_BLOCK_OUT,
                                               vta.VTA_BATCH,
                                               vta.VTA_BLOCK_OUT)
        if check_correctness:
            np.testing.assert_allclose(res_unpack, res_ref)
        return cost

    def run_schedule(load_inp,
                     load_wgt,
                     gemm,
                     alu,
                     store_out,
                     print_ir,
                     check_correctness):
        s = tvm.create_schedule(res.op)
        s[data_buf].set_scope(vta.SCOPE_INP)
        s[weight_buf].set_scope(vta.SCOPE_WGT)
        s[res_gem].set_scope(vta.SCOPE_OUT)
        s[res_shf].set_scope(vta.SCOPE_OUT)
        s[res_min].set_scope(vta.SCOPE_OUT)
        s[res_max].set_scope(vta.SCOPE_OUT)

        if block:
            bblock = block // vta.VTA_BATCH
            iblock = block // vta.VTA_BLOCK_IN
            oblock = block // vta.VTA_BLOCK_OUT
            xbo, xco, xbi, xci = s[res].op.axis
            xb1, xco1, xb2, xco2 = s[res].tile(xbo, xco, bblock, oblock)
            store_pt = xb2

            s[res_gem].compute_at(s[res], xco1)
            s[res_shf].compute_at(s[res], xco1)
            s[res_min].compute_at(s[res], xco1)
            s[res_max].compute_at(s[res], xco1)

            xbo, xco, xbi, xci = s[res_gem].op.axis
            # Compute one line at a time
            ko1, ko2 = s[res_gem].split(ko, iblock)
            s[res_gem].reorder(ko1, ko2, xbo, xco, xbi, xci, ki)
            s[data_buf].compute_at(s[res_gem], ko1)
            s[weight_buf].compute_at(s[res_gem], ko1)
            # Use VTA instructions
            s[data_buf].pragma(s[data_buf].op.axis[0], load_inp)
            s[weight_buf].pragma(s[weight_buf].op.axis[0], load_wgt)
            s[res_gem].tensorize(xbi, gemm)
            s[res_shf].pragma(s[res_shf].op.axis[0], alu)
            s[res_min].pragma(s[res_min].op.axis[0], alu)
            s[res_max].pragma(s[res_max].op.axis[0], alu)
            s[res].pragma(store_pt, store_out)
        else:
            xbo, xco, xbi, xci = s[res_gem].op.axis
            s[res_gem].reorder(ko, xbo, xco, xbi, xci, ki)
            # Use VTA instructions
            s[data_buf].pragma(s[data_buf].op.axis[0], load_inp)
            s[weight_buf].pragma(s[weight_buf].op.axis[0], load_wgt)
            s[res_gem].tensorize(xbi, gemm)
            s[res_shf].pragma(s[res_shf].op.axis[0], alu)
            s[res_min].pragma(s[res_min].op.axis[0], alu)
            s[res_max].pragma(s[res_max].op.axis[0], alu)
            s[res].pragma(s[res].op.axis[0], store_out)

        if print_ir:
            print(tvm.lower(s, [data, weight, res], simple_mode=True))
        return verify(s, check_correctness)

    def gemm_normal(print_ir):
        mock = vta.mock
        print("----- GEMM GFLOPS End-to-End Test-------")
        def run_test(header, print_ir, check_correctness):
            cost = run_schedule(
                vta.DMA_COPY, vta.DMA_COPY, vta.GEMM, vta.ALU, vta.DMA_COPY,
                print_ir, check_correctness)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS" % (cost.mean, gops))
        with tvm.build_config(add_lower_pass=vta.debug_mode(vta.DEBUG_DUMP_INSN)):
            run_test("NORMAL", print_ir, True)

        print("")

    def gevm_unittest(print_ir):
        mock = vta.mock
        print("----- GEMM Unit Test-------")
        def run_test(header, print_ir):
            cost = run_schedule(
                mock.DMA_COPY, mock.DMA_COPY, vta.GEMM, mock.ALU, mock.DMA_COPY,
                print_ir, False)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS" % (cost.mean, gops))
        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir)
        print("")

    def alu_unittest(print_ir):
        mock = vta.mock
        print("----- ALU Unit Test-------")
        def run_test(header, print_ir):
            cost = run_schedule(
                mock.DMA_COPY, mock.DMA_COPY, mock.GEMM, vta.ALU, mock.DMA_COPY,
                print_ir, False)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS" % (cost.mean, gops))
        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir)
        print("")

    def load_inp_unittest(print_ir):
        mock = vta.mock
        print("----- LoadInp Unit Test-------")
        def run_test(header, print_ir):
            cost = run_schedule(
                vta.DMA_COPY, mock.DMA_COPY, mock.GEMM, mock.ALU, mock.DMA_COPY, print_ir, False)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            bandwith = (batch_size * channel * vta.VTA_INP_WIDTH / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS, bandwidth=%g Gbits" % (
                cost.mean, gops, bandwith))
        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir)
        print("")

    def load_wgt_unittest(print_ir):
        mock = vta.mock
        print("----- LoadWgt Unit Test-------")
        def run_test(header, print_ir):
            cost = run_schedule(
                mock.DMA_COPY, vta.DMA_COPY, mock.GEMM, mock.ALU, mock.DMA_COPY, print_ir, False)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            bandwith = (channel * channel * vta.VTA_WGT_WIDTH / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS, bandwidth=%g Gbits" % (
                cost.mean, gops, bandwith))
        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir)
        print("")

    def store_out_unittest(print_ir):
        mock = vta.mock
        print("----- StoreOut Unit Test-------")
        def run_test(header, print_ir):
            cost = run_schedule(
                mock.DMA_COPY, mock.DMA_COPY, mock.GEMM, mock.ALU, vta.DMA_COPY,
                print_ir, False)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            bandwith = (batch_size * channel * vta.VTA_OUT_WIDTH / cost.mean) / float(10 ** 9)
            print(header)
            print("\tTime cost = %g sec/op, %g GFLOPS, bandwidth=%g Gbits" % (
                cost.mean, gops, bandwith))
        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            run_test("NORMAL", print_ir)
        print("")

    gemm_normal(False)
    gevm_unittest(False)
    alu_unittest(False)
    # FIXME: report time that is too short
    # load_inp_unittest(False)
    # load_wgt_unittest(False)
    # store_out_unittest(False)


print("========GEMM 128=========")
test_gemm_packed(128, 128, 128)

# FIXME: hanging run
# print("========GEMM 1024========")
# test_gemm_packed(1024, 1024, 128)
