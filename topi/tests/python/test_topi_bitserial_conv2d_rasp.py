import os
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from tvm.contrib import rpc, util

def generate_quantized_np(shape, bits, out_dtype):
    np.random.seed(0)
    min_val = 0
    max_val = 1 << bits
    return np.random.randint(min_val, max_val, size=shape).astype(out_dtype)

def verify_bitserial_conv2d_nchw(batch, in_size, in_channel, num_filter, kernel, stride, padding, 
                        activation_bits, weight_bits, dorefa):
    target = 'llvm -target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon'
    host = '10.77.1.69'
    port = 9090
    remote = rpc.connect(host, port)
    ctx = remote.cpu(0)  

    in_height = in_width = in_size
    input_type='uint32'
    out_dtype='int32'

    with tvm.target.rasp():
        A = tvm.placeholder((batch, in_channel, in_height, in_width), dtype=input_type, name='A')
        W = tvm.placeholder((num_filter, in_channel, kernel, kernel), dtype=input_type, name='W')
        B = topi.nn.bitserial_conv2d(A, W, stride, padding, activation_bits, weight_bits, out_dtype=out_dtype,
            layout="NCHW", dorefa=dorefa)
        s = topi.generic.schedule_bitserial_conv2d_nchw([B])

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    def get_ref_data():
        a_np = generate_quantized_np(get_const_tuple(A.shape), activation_bits, input_type)
        w_np = generate_quantized_np(get_const_tuple(W.shape), weight_bits, input_type)
        if dorefa:
            w_ = np.copy(w_np).astype(out_dtype)
            for x in np.nditer(w_, op_flags=['readwrite']):
                x[...] = 1 if x == 1 else -1
            b_np = topi.testing.conv2d_nchw_python(a_np, w_, stride, padding).astype(out_dtype)
        else:
            b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding).astype(out_dtype)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, W, B], target)

    # upload to rpi
    temp = util.tempdir()
    path = temp.relpath('conv_nhwc.o')
    func.save(path)
    remote.upload(path)
    func = remote.load_module('conv_nhwc.o')

    func(a, w, b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

def verify_bitserial_conv2d_nhwc(batch, in_size, in_channel, num_filter, kernel, stride, padding, 
                        activation_bits, weight_bits, dorefa):
    target = 'llvm -target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon'
    host = '10.77.1.69'
    port = 9090
    remote = rpc.connect(host, port)
    ctx = remote.cpu(0)  

    in_height = in_width = in_size
    input_type='uint32'
    out_dtype='int32'

    with tvm.target.rasp():
        A = tvm.placeholder((batch, in_height, in_width, in_channel), dtype=input_type, name='A')
        W = tvm.placeholder((kernel, kernel, in_channel, num_filter), dtype=input_type, name='W')
        B = topi.nn.bitserial_conv2d(A, W, stride, padding, activation_bits, weight_bits, out_dtype=out_dtype, 
                            layout="NHWC", dorefa=dorefa)
        s = topi.generic.schedule_bitserial_conv2d_nhwc([B])

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    def get_ref_data():
        a_np = generate_quantized_np(get_const_tuple(A.shape), activation_bits, input_type)
        w_np = generate_quantized_np(get_const_tuple(W.shape), weight_bits, input_type)
        if dorefa:
            w_ = np.copy(w_np).astype(out_dtype)
            for x in np.nditer(w_, op_flags=['readwrite']):
                x[...] = 1 if x == 1 else -1
            b_np = topi.testing.conv2d_nhwc_python(a_np, w_, stride, padding).astype(out_dtype)
        else:
            b_np = topi.testing.conv2d_nhwc_python(a_np, w_np, stride, padding).astype(out_dtype)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, W, B], target)
    # Upload to pi
    temp = util.tempdir()
    path = temp.relpath('conv_nhwc.o')
    func.save(path)
    remote.upload(path)
    func = remote.load_module('conv_nhwc.o')

    func(a, w, b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

def test_bitserial_conv2d():
    in_size = 56
    ic, oc = 64, 64
    k = 3
    stride = 1
    pad = 1
    verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 1, 1, False)
    verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 2, 1, False)
    verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 2, 1, False)
    verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 1, 1, True)
    verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 2, 1, True)

    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 1, 1, False)
    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 1, False)
    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 2, False)

if __name__ == "__main__":
    test_bitserial_conv2d()

