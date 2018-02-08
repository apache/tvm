"""Test code for reduce."""
import os
import numpy as np
import tvm
import topi

def _my_npy_argmax(arr, axis, keepdims):
    if not keepdims:
        return arr.argmax(axis=axis)
    else:
        if axis is not None:
            out_shape = list(arr.shape)
            out_shape[axis] = 1
        else:
            out_shape = [1 for _ in range(len(arr.shape))]
        return arr.argmax(axis=axis).reshape(out_shape)


def _my_npy_argmin(arr, axis, keepdims):
    if not keepdims:
        return arr.argmin(axis=axis)
    else:
        out_shape = list(arr.shape)
        out_shape[axis] = 1
        return arr.argmin(axis=axis).reshape(out_shape)

def verify_reduce_map_ele(in_shape, axis, keepdims, type="sum"):
    # Build the logic and compile the function
    dat_dtype = "float32"
    A = tvm.placeholder(shape=in_shape, name="A", dtype=dat_dtype)
    A1 = topi.cpp.sqrt(topi.cpp.exp(A))
    out_dtype = "float32"
    if type == "sum":
        B = topi.cpp.sum(A1, axis, keepdims)
    elif type == "max":
        B = topi.cpp.max(A1, axis, keepdims)
    elif type == "min":
        B = topi.cpp.min(A1, axis, keepdims)
    elif type == "argmax":
        B = topi.cpp.argmax(A1, axis, keepdims)
        out_dtype = "int32"
    elif type == "argmin":
        B = topi.cpp.argmin(A1, axis, keepdims)
        out_dtype = "int32"
    else:
        raise NotImplementedError

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.default_schedule(target, [B], True)
        else:
            s = topi.cpp.cuda.schedule_reduce(target, [B])

        ctx = tvm.context(device, 0)
        foo = tvm.build(s, [A, B], device, name="sum")
        # Test
        in_npy = np.random.uniform(size=in_shape).astype(np.float32)
        in_npy_map = np.sqrt(np.exp(in_npy)).astype(np.float32)
        if type == "sum":
            out_npy = in_npy_map.sum(axis=axis, keepdims=keepdims)
        elif type == "max":
            out_npy = in_npy_map.max(axis=axis, keepdims=keepdims)
        elif type == "min":
            out_npy = in_npy_map.min(axis=axis, keepdims=keepdims)
        elif type == "argmax":
            out_npy = _my_npy_argmax(in_npy_map, axis=axis, keepdims=keepdims)
        elif type == "argmin":
            out_npy = _my_npy_argmin(in_npy_map, axis=axis, keepdims=keepdims)
        else:
            raise NotImplementedError
        data_tvm = tvm.nd.array(in_npy, ctx=ctx)
        out_tvm = tvm.nd.empty(shape=out_npy.shape, ctx=ctx, dtype=out_dtype)
        for _ in range(1):
            foo(data_tvm, out_tvm)
        if type == "argmax" or type == "argmin":
            out_tvm_indices = out_tvm.asnumpy()
            if keepdims:
                out_tvm_indices = np.take(out_tvm_indices, indices=0, axis=axis)
            if axis is None:
                out_tvm_val = in_npy_map.ravel()[out_tvm_indices]
            else:
                other_indices = tuple(np.indices(in_shape[0:axis] + in_shape[(axis+1):]))
                sel_indices = other_indices[0:axis] + (out_tvm_indices,) + other_indices[axis:]
                out_tvm_val = in_npy_map[sel_indices]
            if type == "argmax":
                np.testing.assert_allclose(out_tvm_val, in_npy_map.max(axis=axis), 1E-3, 1E-3)
            elif type == "argmin":
                np.testing.assert_allclose(out_tvm_val, in_npy_map.min(axis=axis), 1E-3, 1E-3)
        else:
            np.testing.assert_allclose(out_tvm.asnumpy(), out_npy, 1E-3, 1E-3)
    for device in ["cuda", "opencl", "metal", "llvm", "rocm"]:
        check_device(device)


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
    verify_reduce_map_ele(in_shape=(32, 128),
                          axis=1,
                          keepdims=True,
                          type="argmax")
    verify_reduce_map_ele(in_shape=(32, 24, 32, 24),
                          axis=2,
                          keepdims=False,
                          type="argmin")
    verify_reduce_map_ele(in_shape=(31, 21, 15),
                          axis=None,
                          keepdims=True,
                          type="argmax")
    verify_reduce_map_ele(in_shape=(31, 21, 15),
                          axis=None,
                          keepdims=False,
                          type="sum")

if __name__ == "__main__":
    test_reduce_map()
