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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
"""Example code to do convolution."""

import numpy as np
import tvm
import os
import topi
import topi.testing
from tvm import te, autotvm
from tvm.contrib.pickle_memoize import memoize
from tvm.contrib import nvcc
from topi.nn.util import get_pad_tuple
from topi.util import get_const_tuple

TASK="conv_int4"

USE_MANUAL_CODE = False

# @tvm.register_func
# def tvm_callback_cuda_compile(code):
#     ptx =  nvcc.compile_cuda(code, target="ptx")
#     return ptx

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code


_conv2d_nhwc_tensorcore_implement = {
    "cuda": (topi.cuda.conv2d_nhwc_tensorcore_int4, topi.cuda.schedule_conv2d_nhwc_tensorcore_int4)
}


def verify_conv2d_nhwc(batch, in_channel, in_size, num_filter, kernel, stride,
                       padding, dilation=1, add_bias=False, add_relu=False, devices='cuda'):
    """Test the conv2d with tensorcore for nhwc layout"""
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print("Workload: (%d, %d, %d, %d, %d, %d, %d, %d)" % (
        batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation))

    # choose dtype from int4, int8 and float16
    dtype = 'int4'
    out_dtype = 'int32'
    wmma_n = wmma_m = 8
    wmma_k = 32
    in_height = in_width = in_size

    A = te.placeholder((batch, in_height, in_width, in_channel), name='A', dtype=dtype)

    # A = te.placeholder((batch // wmma_m, in_height, in_width, in_channel // wmma_k, wmma_m, wmma_k), name='A', dtype=dtype)
    if dtype == 'int4' or dtype == 'int8':
        # W = te.placeholder((kernel, kernel, num_filter, in_channel), name='W', dtype=dtype)
        W = te.placeholder((kernel, kernel, in_channel // wmma_k, num_filter // wmma_n , wmma_n, wmma_k), name='W', dtype=dtype)
    else:
        W = te.placeholder((kernel, kernel, in_channel, num_filter), name='W', dtype=dtype)

    bias = te.placeholder((1, 1, 1, num_filter), name='bias', dtype=out_dtype)

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    # a_shape = (batch, in_height, in_width, in_channel)
    w_shape = (kernel, kernel, in_channel, num_filter)
    # w_shape = (kernel, kernel, in_channel, num_filter)
    # dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_nhwc.verify_conv2d_nhwc")
    def get_ref_data():
        np.random.seed(5)
        if dtype == 'float16':
            a_np = np.random.uniform(size=a_shape).astype(dtype)
            w_np = np.random.uniform(size=w_shape).astype(dtype)
            b_np = np.random.uniform(size=bias_shape).astype(out_dtype)
            dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        elif dtype == 'int4':
            a_np = np.random.randint(low=1, high=7, size=a_shape).astype(np.int32)
            b_np = np.random.randint(low=1, high=7, size=bias_shape).astype(np.int32)
            w_np = np.random.randint(low=1, high=7, size=w_shape).astype(np.int32)
            dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        elif dtype == 'int8':
            a_np = np.random.randint(low=1, high=7, size=a_shape).astype(dtype)
            w_np = np.random.randint(low=1, high=7, size=w_shape).astype(dtype)
            b_np = np.random.randint(low=1, high=7, size=bias_shape).astype(dtype)
            dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))

        c_np = topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
        if add_bias:
            # b_np = np.random.uniform(size=bias_shape).astype(out_dtype)
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)
        return a_np, w_np, b_np, c_np
        
    def convert_int32_into_int4(a_int32):
        """ convert int32 values into int4
        Parameters
        ----------
        a_int32 : int

        Return
        ------
        a_int4 : int
        """
        I, J, K, L = a_int32.shape
        a_int4 = np.zeros(shape=(I, J, K, L // 8), dtype=np.int32)
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    for l in range(L // 8):
                        for a in range(8):
                            a_int4[i,j,k,l] = a_int4[i,j,k,l] | ((a_int32[i,j,k,l * 8 + a] & 0xf) << ((7 - a) * 4))
        return a_int4
    def convert_int32_into_int4_shape6(a_int32):
        """ convert int32 values into int4
        Parameters
        ----------
        a_int32 : int

        Return
        ------
        a_int4 : int
        """
        M, N, I, J, K, L = a_int32.shape
        a_int4 = np.zeros(shape=(M, N, I, J, K, L // 8), dtype=np.int32)
        for m in range(M):
            for n in range (N):
                for i in range(I):
                    for j in range(J):
                        for k in range(K):
                            for l in range(L // 8):
                                for a in range(8):
                                    a_int4[m,n,i,j,k,l] = a_int4[m,n,i,j,k,l] | ((a_int32[m,n,i,j,k,l * 8 + a] & 0xf) << ((7 - a) * 4))
        return a_int4

    a_np, w_np, b_np, c_np = get_ref_data()

    if dtype == 'int4':
        # a_np_tvm = a_np.reshape((batch // wmma_m,
        #         wmma_m,
        #         in_height,
        #         in_width,
        #         in_channel // wmma_k,
        #         wmma_k)).transpose((0,2,3,4,1,5))
        w_np = w_np.reshape((kernel,
                    kernel,
                    in_channel // wmma_k,
                    wmma_k,
                    num_filter // wmma_n,
                    wmma_n)).transpose((0,1,2,4,5,3))
        a_np = convert_int32_into_int4(a_np)
        # b_np = convert_int32_into_int4(b_np)
        w_np = convert_int32_into_int4_shape6(w_np)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        if not nvcc.have_tensorcore(ctx.compute_version):
            print("skip because gpu does not support Tensor Cores")
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            fcompute, fschedule = topi.testing.dispatch(device, _conv2d_nhwc_tensorcore_implement)
            if dtype == 'float16':
                C = fcompute(A, W, stride, padding, dilation, dtype, 'float')
            else:
                C = fcompute(A, W, stride, padding, dilation, dtype, 'int32')
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = fschedule([C])
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        if add_bias:
            func = tvm.build(s, [A, W, bias, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (
                batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation))
            func(a, w, b, c)
        else:
            print(tvm.lower(s, [A, W, C], simple_mode=True))
            func = tvm.build(s, [A, W, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (
                batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation))
            func(a, w, c)
            dev_module = func.imported_modules[0]
            # print(dev_module.get_source())
            # warm up
            evaluator = func.time_evaluator(func.entry_name, ctx, number=50, repeat=20)
            evaluator(a, w, c)
            print('Time cost of this operator: %f ms' % (evaluator(a, w, c).mean * 1000))

        rtol = 1e-3
        # print(c.asnumpy().sum(), c_np.sum())
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=rtol)

        # # #Tuning the performance 
        # import logging, sys
        # logging.getLogger('autotvm').setLevel(logging.DEBUG)
        # logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

        # log_filename = "conv2d_int4_nhwc_tensorcore_injectpad_%d_%d_%d_%d_%d_%d_%d_%d.log" % (batch, in_channel, in_size, num_filter, kernel, stride,
        #                padding, dilation)
        # tmp_log_file = log_filename + '.temp'
        # num_trial = 2000
        # task_name = "conv2d_nhwc_tensorcore_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride,
        #                padding, dilation)
        # task = autotvm.create('conv2d_nhwc_tensorcore_int4.cuda', args=[A, W, stride, padding, dilation, dtype, out_dtype], target=device)
        # print(task.config_space)
        
        # measure_option = autotvm.measure_option(
        #     builder='local',
        #     runner=autotvm.LocalRunner(number=5))

        # tuner = autotvm.tuner.XGBTuner(task)
        # num_trial = min(num_trial, len(task.config_space))
        # with tvm.target.build_config():
        #     tuner.tune(n_trial=num_trial,
        #                measure_option=measure_option,
        #                callbacks=[autotvm.callback.progress_bar(num_trial, prefix=task_name),
        #                           autotvm.callback.log_to_file(tmp_log_file)])

        # dispatch_context = autotvm.apply_history_best(tmp_log_file)
        # best_config = dispatch_context.query(task.target, task.workload)
        # print("\nBest config:")
        # print(best_config)

        # #pick the best record to a cache file
        # autotvm.record.pick_best(tmp_log_file, log_filename)
        # os.remove(tmp_log_file)
        
        # with autotvm.apply_graph_best(log_filename):
        #     with tvm.target.create(device):
        #         func = tvm.build(s, [A, W, C], device, name="conv2d_nhwc_tensorcore_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride,
        #                padding, dilation))
        #         evaluator = func.time_evaluator(func.entry_name, ctx, number=50, repeat=20)
        #         print('Time cost of this operator after tuning: %f ms' % (evaluator(a, w, c).mean * 1000))

    check_device(devices)


def test_conv2d_nhwc_tensorcore():
    """Test the conv2d with tensorcore for nhwc layout"""
    # verify_conv2d_nhwc(64, 64, 56, 64, 3, 1, 1)
    # verify_conv2d_nhwc(64, 64, 56, 64, 1, 1, 0)
    # verify_conv2d_nhwc(64, 64, 56, 128, 3, 2, 1)
    # verify_conv2d_nhwc(64, 64, 56, 64, 1, 2, 0)
    # verify_conv2d_nhwc(64, 128, 28, 128, 3, 1, 1)
    # verify_conv2d_nhwc(64, 128, 28, 256, 3, 2, 1)
    # verify_conv2d_nhwc(64, 128, 28, 256, 1, 2, 0)
    # verify_conv2d_nhwc(64, 256, 14, 256, 3, 1, 1)
    # verify_conv2d_nhwc(64, 256, 14, 512, 3, 2, 1)
    # verify_conv2d_nhwc(64, 256, 14, 512, 1, 2, 0)
    # verify_conv2d_nhwc(64, 512, 7, 512, 3, 1, 1)

    # verify_conv2d_nhwc(32, 64, 56, 64, 3, 1, 1)
    # verify_conv2d_nhwc(32, 64, 56, 64, 1, 1, 0)
    # verify_conv2d_nhwc(32, 64, 56, 128, 3, 2, 1)
    # verify_conv2d_nhwc(32, 64, 56, 64, 1, 2, 0)
    # verify_conv2d_nhwc(32, 128, 28, 128, 3, 1, 1)
    # verify_conv2d_nhwc(32, 128, 28, 256, 3, 2, 1)
    # verify_conv2d_nhwc(32, 128, 28, 256, 1, 2, 0)
    # verify_conv2d_nhwc(32, 256, 14, 256, 3, 1, 1)
    # verify_conv2d_nhwc(32, 256, 14, 512, 3, 2, 1)
    # verify_conv2d_nhwc(32, 256, 14, 512, 1, 2, 0)
    # verify_conv2d_nhwc(32, 512, 7, 512, 3, 1, 1)

    # verify_conv2d_nhwc(16, 64, 56, 64, 3, 1, 1)
    # verify_conv2d_nhwc(16, 64, 56, 64, 1, 1, 0)
    # verify_conv2d_nhwc(16, 64, 56, 128, 3, 2, 1)
    # verify_conv2d_nhwc(16, 64, 56, 64, 1, 2, 0)
    # verify_conv2d_nhwc(16, 128, 28, 128, 3, 1, 1)
    # verify_conv2d_nhwc(16, 128, 28, 256, 3, 2, 1)
    # verify_conv2d_nhwc(16, 128, 28, 256, 1, 2, 0)
    # verify_conv2d_nhwc(16, 256, 14, 256, 3, 1, 1)
    # verify_conv2d_nhwc(16, 256, 14, 512, 3, 2, 1)
    # verify_conv2d_nhwc(16, 256, 14, 512, 1, 2, 0)
    # verify_conv2d_nhwc(16, 512, 7, 512, 3, 1, 1)

    verify_conv2d_nhwc(8, 64, 56, 64, 3, 1, 1)
    verify_conv2d_nhwc(8, 64, 56, 64, 1, 1, 0)
    verify_conv2d_nhwc(8, 64, 56, 128, 3, 2, 1)
    verify_conv2d_nhwc(8, 64, 56, 64, 1, 2, 0)
    verify_conv2d_nhwc(8, 128, 28, 128, 3, 1, 1)
    verify_conv2d_nhwc(8, 128, 28, 256, 3, 2, 1)
    verify_conv2d_nhwc(8, 128, 28, 256, 1, 2, 0)
    verify_conv2d_nhwc(8, 256, 14, 256, 3, 1, 1)
    verify_conv2d_nhwc(8, 256, 14, 512, 3, 2, 1)
    verify_conv2d_nhwc(8, 256, 14, 512, 1, 2, 0)
    verify_conv2d_nhwc(8, 512, 7, 512, 3, 1, 1)


    # verify_conv2d_nhwc(32, 1024, 14, 256, 1, 1, 1)

    # verify_conv2d_nhwc(16, 128, 7, 128, 7, 1, 3)
    # verify_conv2d_nhwc(16, 160, 7, 160, 7, 1, 3)

    # verify_conv2d_nhwc(32, 64, 14, 64, 3, 1, 1, add_bias=True)
    # verify_conv2d_nhwc(32, 64, 14, 64, 3, 1, 1, add_relu=True)
    # verify_conv2d_nhwc(32, 64, 14, 64, 3, 1, 1, add_relu=True, add_bias=True)

    # verify_conv2d_nhwc(16, 64, 17, 64, 7, 1, (3, 3, 2, 2))
    # verify_conv2d_nhwc(16, 64, 17, 64, 7, 1, "SAME")
    # verify_conv2d_nhwc(16, 48, 35, 48, 5, 1, "VALID")
    # verify_conv2d_nhwc(16, 48, 56, 48, 3, 1, (1, 1, 1, 1))
    # verify_conv2d_nhwc(16, 64, 28, 64, 3, 1, (1, 1, 1, 1))


if __name__ == "__main__":
    test_conv2d_nhwc_tensorcore()
