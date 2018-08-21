import os
import re
import numpy as np
import tvm
import topi
import topi.testing

def generate_quantized_np(shape, bits, out_dtype):
    np.random.seed(0)
    min_val = 0
    max_val = 1 << bits
    return np.random.randint(min_val, max_val, size=shape).astype(out_dtype)

# Verify that certain special instructions from the tensorize pass exist
def verify_bitserial_conv2d_nhwc(batch, in_size, in_channel, num_filter, kernel, stride, padding, 
                                 activation_bits, weight_bits, dorefa):
    in_height = in_width = in_size
    input_type = 'uint32'
    out_dtype = 'int32'

    with tvm.target.arm_cpu('rasp3b'):
        A = tvm.placeholder((batch, in_height, in_width, in_channel), dtype=input_type, name='A')
        W = tvm.placeholder((kernel, kernel, in_channel, num_filter), dtype=input_type, name='W')
        B = topi.nn.bitserial_conv2d(A, W, stride, padding, activation_bits, weight_bits, out_dtype=out_dtype, 
                                     layout="NHWC", dorefa=dorefa)
        s = topi.generic.schedule_bitserial_conv2d_nhwc([B])

    func = tvm.build(s, [A, W, B], tvm.target.arm_cpu('rasp3b'))
   
    assembly = func.get_source('asm')
    matches = re.findall("vpadal", assembly)
    assert (len(matches) > 0)
    matches = re.findall("vcnt", assembly)
    assert (len(matches) > 0)
    matches = re.findall("vpadd", assembly)
    assert (len(matches) > 0)

def test_bitserial_conv2d():
    in_size = 56
    ic, oc = 64, 64
    k = 3
    stride = 1
    pad = 1

    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 1, 1, False)
    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 1, False)

if __name__ == "__main__":
    test_bitserial_conv2d()

