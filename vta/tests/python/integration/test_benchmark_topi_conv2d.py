"""Testing if we can generate code in topi style"""

import tvm
from tvm import autotvm
from tvm.contrib import util
from tvm.contrib.pickle_memoize import memoize
import topi
import topi.testing
import vta
import vta.testing
import numpy as np

Workload = vta.top.vta_conv2d.Workload

@tvm.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    x = tvm.compute(x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
    x = tvm.compute(x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return x

def test_cpu_conv2d():
    def run_cpu_conv2d(env, remote, key, batch_size, wl, profile=True):
        data_shape = (batch_size, wl.in_filter, wl.height, wl.width)
        kernel_shape = (wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)

        fout_height = (wl.height + 2 * wl.hpad - wl.hkernel) // wl.hstride + 1
        fout_width = (wl.width + 2 * wl.wpad - wl.wkernel) // wl.wstride + 1
        data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
        kernel = tvm.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
        res_conv = topi.nn.conv2d(
            data, kernel, padding=(wl.hpad, wl.wpad),
            strides=(wl.hstride, wl.wstride),
            out_dtype="int32")
        res = topi.right_shift(res_conv, 8)
        res = my_clip(res, 0, 127)
        res = topi.cast(res, "int8")

        # To compute number of ops, use a x2 factor for FMA
        num_ops = 2 * batch_size * fout_height * fout_width * wl.hkernel * wl.wkernel * wl.out_filter * wl.in_filter

        a_shape = (batch_size, wl.in_filter, wl.height, wl.width)
        w_shape = (wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)
        stride = (wl.hstride, wl.wstride)
        data_dtype = data.dtype
        kernel_dtype = kernel.dtype
        acc_dtype = env.acc_dtype
        assert wl.hpad == wl.wpad
        padding = wl.hpad

        @memoize("vta.tests.test_benchmark_topi.conv2d.cpu.verify_nhwc")
        def get_ref_data():
            a_np = (np.random.uniform(size=a_shape) * 4).astype(data_dtype)
            w_np = (np.random.uniform(size=w_shape) * 4).astype(kernel_dtype)
            a_np = np.abs(a_np)
            w_np = np.abs(w_np)
            b_np = topi.testing.conv2d_nchw_python(
                a_np.astype(acc_dtype), w_np.astype(acc_dtype), stride, padding).astype(acc_dtype)
            return a_np, w_np, b_np


        def verify(s, check_correctness):
            mod = tvm.build(s, [data, kernel, res],
                            target_host=env.target_host,
                            name="conv2d")
            temp = util.tempdir()
            mod.save(temp.relpath("conv2d.o"))
            remote.upload(temp.relpath("conv2d.o"))
            f = remote.load_module("conv2d.o")
            # verify
            ctx = remote.cpu(0)
            # Data in original format
            data_orig, kernel_orig, res_ref = get_ref_data()
            res_shape = topi.util.get_const_tuple(res.shape)
            res_np = np.zeros(res_shape).astype(res.dtype)
            data_arr = tvm.nd.array(data_orig, ctx)
            kernel_arr = tvm.nd.array(kernel_orig, ctx)
            res_arr = tvm.nd.array(res_np, ctx)
            time_f = f.time_evaluator("conv2d", ctx, number=5)
            cost = time_f(data_arr, kernel_arr, res_arr)
            res_unpack = res_arr.asnumpy()
            if check_correctness:
                assert wl.hpad == wl.wpad
                stride = (wl.hstride, wl.wstride)
                padding = wl.hpad
                res_ref = res_ref >> 8
                res_ref = np.clip(res_ref, 0, 127).astype("int8")
                np.testing.assert_allclose(res_unpack, res_ref)
            return cost

        def conv_normal(print_ir):
            print("----- CONV2D CPU End-to-End Test-------")
            s = topi.generic.schedule_conv2d_nchw([res])
            if print_ir:
                print(tvm.lower(s, [data, kernel, res], simple_mode=True))
            cost = verify(s, True)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            print("\tTime cost = %g sec/op, %g GOPS" % (cost.mean, gops))

        conv_normal(False)

    def _run(env, remote):
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
        batch_size = 1
        for i in range(1, len(resnet)):
            wl = resnet[i]
            key = "resnet-cfg[%d]" % i
            print("key=%s" % key)
            print(wl)
            with tvm.target.create("llvm -device=vtacpu"):
                run_cpu_conv2d(env, remote, key, batch_size, wl)

    # load pre-tuned operator parameters for ARM CPU
    autotvm.tophub.check_backend('vta')
    with autotvm.tophub.context('llvm -device=vtacpu'):
        vta.testing.run(_run)


def test_vta_conv2d():
    def run_vta_conv2d(env, remote, key, batch_size, wl, profile=True):
        data_shape = (batch_size//env.BATCH, wl.in_filter//env.BLOCK_IN,
                      wl.height, wl.width, env.BATCH, env.BLOCK_IN)
        kernel_shape = (wl.out_filter//env.BLOCK_OUT, wl.in_filter//env.BLOCK_IN,
                        wl.hkernel, wl.wkernel, env.BLOCK_OUT, env.BLOCK_IN)
        bias_shape = (1, wl.out_filter//env.BLOCK_OUT, 1, 1, env.BATCH, env.BLOCK_OUT)

        fout_height = (wl.height + 2 * wl.hpad - wl.hkernel) // wl.hstride + 1
        fout_width = (wl.width + 2 * wl.wpad - wl.wkernel) // wl.wstride + 1
        data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
        kernel = tvm.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
        bias = tvm.placeholder(bias_shape, name="kernel", dtype=env.acc_dtype)

        res_conv = vta.top.packed_conv2d(
            data, kernel, padding=(wl.hpad, wl.wpad), strides=(wl.hstride, wl.wstride))
        res = topi.right_shift(res_conv, 8)
        res = topi.add(res, bias)
        res = my_clip(res, 0, 127)
        res = topi.cast(res, "int8")

        # To compute number of ops, use a x2 factor for FMA
        num_ops = 2 * batch_size * fout_height * fout_width * wl.hkernel * wl.wkernel * wl.out_filter * wl.in_filter

        a_shape = (batch_size, wl.in_filter, wl.height, wl.width)
        w_shape = (wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)
        stride = (wl.hstride, wl.wstride)
        data_dtype = data.dtype
        kernel_dtype = kernel.dtype
        acc_dtype = env.acc_dtype
        assert wl.hpad == wl.wpad
        padding = wl.hpad

        @memoize("vta.tests.test_benchmark_topi.conv2d.verify_nhwc")
        def get_ref_data():
            a_np = (np.random.uniform(size=a_shape) * 4).astype(data_dtype)
            w_np = (np.random.uniform(size=w_shape) * 4).astype(kernel_dtype)
            a_np = np.abs(a_np)
            w_np = np.abs(w_np)
            b_np = topi.testing.conv2d_nchw_python(
                a_np.astype(acc_dtype), w_np.astype(acc_dtype), stride, padding).astype(acc_dtype)
            return a_np, w_np, b_np

        def verify(s, check_correctness):
            mod = vta.build(s, [data, kernel, bias, res], "ext_dev",
                            env.target_host, name="conv2d")
            temp = util.tempdir()

            mod.save(temp.relpath("conv2d.o"))
            remote.upload(temp.relpath("conv2d.o"))
            f = remote.load_module("conv2d.o")
            # verify
            ctx = remote.ext_dev(0)
            # Data in original format
            data_orig, kernel_orig, res_ref = get_ref_data()
            bias_orig = (np.random.uniform(size=(wl.out_filter,)) * 4).astype("int32")
            bias_orig = np.abs(bias_orig)

            data_packed = data_orig.reshape(
                batch_size//env.BATCH, env.BATCH,
                wl.in_filter//env.BLOCK_IN, env.BLOCK_IN,
                wl.height, wl.width).transpose((0, 2, 4, 5, 1, 3))
            kernel_packed = kernel_orig.reshape(
                wl.out_filter//env.BLOCK_OUT, env.BLOCK_OUT,
                wl.in_filter//env.BLOCK_IN, env.BLOCK_IN,
                wl.hkernel, wl.wkernel).transpose((0, 2, 4, 5, 1, 3))
            bias_packed = bias_orig.reshape(
                1, wl.out_filter // env.BLOCK_OUT, 1, 1, env.BATCH, env.BLOCK_OUT)
            res_shape = topi.util.get_const_tuple(res.shape)

            res_np = np.zeros(res_shape).astype(res.dtype)
            data_arr = tvm.nd.array(data_packed, ctx)
            kernel_arr = tvm.nd.array(kernel_packed, ctx)
            bias_arr = tvm.nd.array(bias_packed, ctx)
            res_arr = tvm.nd.array(res_np, ctx)
            time_f = f.time_evaluator("conv2d", ctx, number=5)
            cost = time_f(data_arr, kernel_arr, bias_arr, res_arr)
            res_unpack = res_arr.asnumpy().transpose(
                (0, 4, 1, 5, 2, 3)).reshape(batch_size, wl.out_filter, fout_height, fout_width)
            if check_correctness:
                assert wl.hpad == wl.wpad
                stride = (wl.hstride, wl.wstride)
                padding = wl.hpad
                res_ref = res_ref >> 8
                res_ref += bias_orig.reshape(wl.out_filter, 1, 1)
                res_ref = np.clip(res_ref, 0, 127).astype("int8")
                np.testing.assert_allclose(res_unpack, res_ref)
            return cost

        def conv_normal(print_ir):
            print("----- CONV2D End-to-End Test-------")
            with vta.build_config():
                s = vta.top.schedule_packed_conv2d([res])
                if print_ir:
                    print(vta.lower(s, [data, kernel, bias, res], simple_mode=True))
            cost = verify(s, True)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            print("\tTime cost = %g sec/op, %g GOPS" % (cost.mean, gops))

        conv_normal(False)

    def _run(env, remote):
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

        batch_size = 1
        for i in range(0, len(resnet)):
            wl = resnet[i]
            key = "resnet-cfg[%d]" % i
            print("key=%s" % key)
            print(wl)
            run_vta_conv2d(env, remote, key, batch_size, wl)

    vta.testing.run(_run)


if __name__ == "__main__":
    test_cpu_conv2d()
    test_vta_conv2d()
