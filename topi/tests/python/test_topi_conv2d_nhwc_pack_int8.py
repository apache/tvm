"""Example code to do convolution."""
import os
import numpy as np
import tvm
from tvm import autotvm
from tvm.autotvm.task.space import FallbackConfigEntity
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple


def verify_conv2d_1x1_nhwc_pack_int8(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A', dtype='uint8')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W', dtype='int8')

    a_shape = get_const_tuple(A.shape)
    w_shape = (kernel, kernel, in_channel, num_filter)
    dtype = A.dtype

    #@memoize("topi.tests.test_topi_conv2d_1x1_nhwc_pack_int8.verify_nhwc.v2")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(W.dtype)
        dw_np = topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
        b_np = topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        with tvm.target.create(device):
            B = topi.nn.conv2d(A, W, stride, padding, dilation, layout='NHWC', out_dtype="int32")
            s = topi.generic.schedule_conv2d_nhwc([B])
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func = tvm.build(s, [A, W, B], device)
        func(a, w, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm -mcpu=skylake-avx512']:
        check_device(device)


class DefaultFallback(autotvm.FallbackContext):
    def _query_inside(self, target, workload):
        key = (target, workload)
        if key in self.memory:
            return self.memory[key]
        cfg = FallbackConfigEntity()
        cfg.template_key = 'direct'
        self.memory[key] = cfg
        return cfg


def test_conv2d_nhwc():
    autotvm.DispatchContext.current.silent = True
    with DefaultFallback():
        verify_conv2d_1x1_nhwc_pack_int8(1, 256, 32, 256, 1, 1, 0)


if __name__ == "__main__":
    test_conv2d_nhwc()
