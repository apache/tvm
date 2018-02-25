"""Example code to do shortcut."""
import os
import numpy as np
import tvm
import topi
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple


def verify_shortcut(batch, in_size, in_channel):
    in_height = in_width = in_size

    with tvm.target.rasp():
        A1 = tvm.placeholder((batch, in_channel, in_height, in_width), name='A1')
        A2 = tvm.placeholder((batch, in_channel, in_height, in_width), name='A2')
        B = topi.nn.vision.shortcut(A1, A2)
        s = topi.generic.schedule_shortcut([B])

    a_shape = get_const_tuple(A1.shape)
    dtype = A1.dtype

    @memoize("topi.tests.test_topi_reorg.verify_reorg")
    def get_ref_data_shortcut():
        a_np1 = np.random.uniform(size=a_shape).astype(dtype)
        a_np2 = np.random.uniform(size=a_shape).astype(dtype)
        b_np = topi.testing.shortcut_python(a_np1, a_np2)
        return a_np1, a_np2, b_np

    a_np1, a_np2, b_np = get_ref_data_shortcut()
    ctx = tvm.cpu(0)
    a1 = tvm.nd.array(a_np1, ctx)
    a2 = tvm.nd.array(a_np2, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A1,A2, B], "llvm")
    func(a1,a2, b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

def test_shortcut():
    verify_shortcut(1, 144, 32)

if __name__ == "__main__":
    test_shortcut()
