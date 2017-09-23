"""Test code for reduce."""
import os
import numpy as np
import tvm
import topi

def verify_reduce_map_ele(in_shape, axis, keepdims, type="sum"):
    # Build the logic and compile the function
    A = tvm.placeholder(shape=in_shape, name="A")
    A1 = topi.sqrt(topi.exp(A))
    if type == "sum":
        B = topi.sum(A1, axis=axis, keepdims=keepdims)
    elif type == "max":
        B = topi.max(A1, axis=axis, keepdims=keepdims)
    elif type == "min":
        B = topi.min(A1, axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError
    s = topi.cuda.schedule_reduce(B)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        foo = tvm.build(s, [A, B], device, name="sum")
        # Test
        in_npy = np.random.uniform(size=in_shape).astype(np.float32)
        in_npy_map = np.sqrt(np.exp(in_npy))
        if type == "sum":
            out_npy = in_npy_map.sum(axis=axis, keepdims=keepdims)
        elif type == "max":
            out_npy = in_npy_map.max(axis=axis, keepdims=keepdims)
        elif type == "min":
            out_npy = in_npy_map.min(axis=axis, keepdims=keepdims)
        else:
            raise NotImplementedError

        data_tvm = tvm.nd.array(in_npy, ctx=ctx)
        out_tvm = tvm.nd.empty(shape=out_npy.shape, ctx=ctx)
        for _ in range(1):
            foo(data_tvm, out_tvm)
        np.testing.assert_allclose(out_tvm.asnumpy(), out_npy, 1E-3, 1E-3)

    check_device("opencl")
    check_device("cuda")
    check_device("metal")


def test_reduce_map():
    verify_reduce_map_ele(in_shape=(128, 24, 128, 24),
                        axis=(1, 2, 3),
                        keepdims=True,
                        type="sum")
    verify_reduce_map_ele(in_shape=(128, 24 * 128 * 24),
                        axis=(1,),
                        keepdims=False,
                        type="max")
    verify_reduce_map_ele(in_shape=(32, 128, 24),
                        axis=None,
                        keepdims=True,
                        type="sum")
    verify_reduce_map_ele(in_shape=(128, 24, 128, 24),
                        axis=(0, 2),
                        keepdims=False,
                        type="min")

if __name__ == "__main__":
    test_reduce_map()
