"""Testing if we can generate code in topi style"""

import topi
import tvm
from tvm.contrib import util, rpc
import vta
from vta import vta_conv2d
import numpy as np
import mxnet as mx

Workload = vta_conv2d.Workload

@tvm.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    x = tvm.compute(x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
    x = tvm.compute(x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return x

host = "pynq"
port = 9091
out_dtype = "int%d" % vta.VTA_OUT_WIDTH
wgt_dtype = "int%d" % vta.VTA_WGT_WIDTH
inp_dtype = "int%d" % vta.VTA_INP_WIDTH
target = "llvm -target=armv7-none-linux-gnueabihf -mattr=+neon"
print_ir = False


def test_vta_conv2d(key, batch_size, wl, profile=True):
    data_shape = (batch_size, wl.in_filter//vta.VTA_BLOCK_IN,
                  wl.height, wl.width, vta.VTA_BLOCK_IN)
    kernel_shape = (wl.out_filter//vta.VTA_BLOCK_OUT, wl.in_filter//vta.VTA_BLOCK_IN,
                    wl.hkernel, wl.wkernel, vta.VTA_BLOCK_OUT, vta.VTA_BLOCK_IN)
    bias_shape = (wl.out_filter//vta.VTA_BLOCK_OUT, 1, 1, vta.VTA_BLOCK_OUT)


    fout_height = (wl.height + 2 * wl.hpad - wl.hkernel) // wl.hstride + 1
    fout_width = (wl.width + 2 * wl.wpad - wl.wkernel) // wl.wstride + 1
    data = tvm.placeholder(data_shape, name="data", dtype=inp_dtype)
    kernel = tvm.placeholder(kernel_shape, name="kernel", dtype=wgt_dtype)
    bias = tvm.placeholder(bias_shape, name="kernel", dtype=out_dtype)

    res_conv = vta_conv2d.packed_conv2d(
        data, kernel, padding=(wl.hpad, wl.wpad), strides=(wl.hstride, wl.wstride))
    res = topi.right_shift(res_conv, 8)
    res = topi.broadcast_add(res, bias)
    res = my_clip(res, 0, 127)
    res = topi.cast(res, "int8")

    num_ops = fout_height * fout_width * wl.hkernel * wl.wkernel * wl.out_filter * wl.in_filter

    def verify(s, check_correctness):
        mod = tvm.build(s, [data, kernel, bias, res], "ext_dev", target, name="conv2d")
        temp = util.tempdir()
        remote = rpc.connect(host, port)

        mod.save(temp.relpath("conv2d.o"))
        remote.upload(temp.relpath("conv2d.o"))
        f = remote.load_module("conv2d.o")
        # verify
        ctx = remote.ext_dev(0)
        # Data in original format
        data_orig = (np.random.uniform(
            size=(batch_size, wl.in_filter, wl.height, wl.width)) * 4).astype(data.dtype)
        kernel_orig = (np.random.uniform(
            size=(wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)) * 4).astype(kernel.dtype)
        bias_orig = (np.random.uniform(size=(wl.out_filter,)) * 4).astype("int32")

        data_orig = np.abs(data_orig)
        kernel_orig = np.abs(kernel_orig)
        bias_orig = np.abs(bias_orig)

        data_packed = data_orig.reshape(
            batch_size, wl.in_filter//vta.VTA_BLOCK_IN, vta.VTA_BLOCK_IN,
            wl.height, wl.width).transpose((0, 1, 3, 4, 2))
        kernel_packed = kernel_orig.reshape(
            wl.out_filter//vta.VTA_BLOCK_OUT, vta.VTA_BLOCK_OUT,
            wl.in_filter//vta.VTA_BLOCK_IN, vta.VTA_BLOCK_IN,
            wl.hkernel, wl.wkernel).transpose((0, 2, 4, 5, 1, 3))
        bias_packed = bias_orig.reshape(
            wl.out_filter//vta.VTA_BLOCK_OUT, 1, 1, vta.VTA_BLOCK_OUT)
        res_shape = topi.util.get_const_tuple(res.shape)

        res_np = np.zeros(res_shape).astype(res.dtype)
        data_arr = tvm.nd.array(data_packed, ctx)
        kernel_arr = tvm.nd.array(kernel_packed, ctx)
        bias_arr = tvm.nd.array(bias_packed, ctx)
        res_arr = tvm.nd.array(res_np, ctx)
        time_f = f.time_evaluator("conv2d", ctx, number=10)
        cost = time_f(data_arr, kernel_arr, bias_arr, res_arr)
        res_unpack = res_arr.asnumpy().transpose(
            (0, 1, 4, 2, 3)).reshape(batch_size, wl.out_filter, fout_height, fout_width)
        if check_correctness:
            res_ref = mx.nd.Convolution(
                mx.nd.array(data_orig.astype(out_dtype), mx.cpu(0)),
                mx.nd.array(kernel_orig.astype(out_dtype), mx.cpu(0)),
                stride=(wl.hstride, wl.wstride),
                kernel=(wl.hkernel, wl.wkernel),
                num_filter=wl.out_filter,
                no_bias=True,
                pad=(wl.hpad, wl.wpad)).asnumpy().astype(out_dtype)
            res_ref = res_ref >> 8
            res_ref += bias_orig.reshape(wl.out_filter, 1, 1)
            res_ref = np.clip(res_ref, 0, 127).astype("int8")
            np.testing.assert_allclose(res_unpack, res_ref)
            print("Correctness check pass...")
        return cost

    def conv_normal(print_ir):
        print("----- CONV2D End-to-End Test-------")
        with tvm.build_config(add_lower_pass=vta.debug_mode(0)):
            s = vta_conv2d.schedule_packed_conv2d([res])
            if print_ir:
                print(tvm.lower(s, [data, kernel, bias, res], simple_mode=True))
            cost = verify(s, True)
        gops = (num_ops / cost.mean) / float(10 ** 9)
        print("\tTime cost = %g sec/op, %g GFLOPS" % (cost.mean, gops))

    conv_normal(print_ir)

# ResNet18 workloads
resnet = {
    # Workloads of resnet18 on imagenet
    0: Workload(224, 224, 16, 64, 7, 7, 3, 3, 2, 2),
    1: Workload(56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
    2: Workload(56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
    3: Workload(56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
    4: Workload(56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
    5: Workload(28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
    6: Workload(28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
    7: Workload(28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
    8: Workload(14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
    9: Workload(14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
    10: Workload(14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
    11: Workload(7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
}

batch_size = 1
for i in range(0, len(resnet)):
    wl = resnet[i]
    key = "resnet-cfg[%d]" % i
    print "key=%s" % key
    print wl
    test_vta_conv2d(key, batch_size, wl)
