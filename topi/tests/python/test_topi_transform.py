"""Test code for broadcasting operators."""
import numpy as np
import tvm
import topi

def verify_expand_dims(in_shape, out_shape, axis, num_newaxis):
    A = tvm.placeholder(shape=in_shape, name="A")
    B = topi.expand_dims(A, axis, num_newaxis)
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        with tvm.target.create(device):
            s = topi.generic.schedule_broadcast(B)
        ctx = tvm.context(device, 0)
        foo = tvm.build(s, [A, B], device, name="expand_dims")
        data_npy = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = data_npy.reshape(out_shape)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nd = tvm.nd.array(np.empty(out_shape).astype(B.dtype), ctx)
        foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    check_device("opencl")
    check_device("cuda")
    check_device("metal")
    check_device("rocm")


def verify_tranpose(in_shape, axes):
    A = tvm.placeholder(shape=in_shape, name="A")
    B = topi.transpose(A, axes)
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(B)
        ctx = tvm.context(device, 0)
        foo = tvm.build(s, [A, B], device, name="tranpose")
        data_npy = np.arange(np.prod(in_shape)).reshape(in_shape).astype(A.dtype)
        out_npy = data_npy.transpose(axes)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nd = tvm.nd.empty(out_npy.shape, ctx=ctx, dtype=B.dtype)
        foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    check_device("cuda")
    check_device("opencl")
    check_device("metal")
    check_device("rocm")

def verify_reshape(src_shape, dst_shape):
    A = tvm.placeholder(shape=src_shape, name="A")
    B = topi.reshape(A, dst_shape)
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(B)
        ctx = tvm.context(device, 0)
        foo = tvm.build(s, [A, B], device, name="reshape")
        data_npy = np.random.normal(size=src_shape).astype(A.dtype)
        out_npy = np.reshape(data_npy, newshape=dst_shape)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nd = tvm.nd.empty(dst_shape, ctx=ctx, dtype=B.dtype)
        foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    check_device("cuda")
    check_device("opencl")
    check_device("metal")
    check_device("rocm")

def verify_squeeze(src_shape, axis):
    A = tvm.placeholder(shape=src_shape, name="A")
    B = topi.squeeze(A, axis=axis)
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(B)
        ctx = tvm.context(device, 0)
        foo = tvm.build(s, [A, B], device, name="squeeze")
        data_npy = np.random.normal(size=src_shape).astype(A.dtype)
        out_npy = np.squeeze(data_npy, axis=axis)
        data_nd = tvm.nd.array(data_npy, ctx)
        if out_npy.shape == ():
            out_nd_shape = (1,)
        else:
            out_nd_shape = out_npy.shape
        out_nd = tvm.nd.empty(out_nd_shape, ctx=ctx, dtype=B.dtype)
        foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    check_device("cuda")
    check_device("opencl")
    check_device("metal")
    check_device("rocm")

def verify_concatenate(shapes, axis):
    tensor_l = []
    for i, shape in enumerate(shapes):
        tensor_l.append(tvm.placeholder(shape, name="A" + str(i)))
    out_tensor = topi.concatenate(a_tuple=tensor_l, axis=axis)
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(out_tensor)
        ctx = tvm.context(device, 0)
        foo = tvm.build(s, tensor_l + [out_tensor], device, name="concatenate")
        data_npys = [np.random.normal(size=shape).astype(tensor_l[0].dtype) for shape in shapes]
        out_npy = np.concatenate(data_npys, axis=axis)
        data_nds = [tvm.nd.array(data_npy, ctx) for data_npy in data_npys]
        out_nd = tvm.nd.empty(out_npy.shape, ctx=ctx, dtype=out_tensor.dtype)
        foo(*(data_nds + [out_nd]))
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    check_device("cuda")
    check_device("opencl")
    check_device("metal")
    check_device("rocm")

def verify_split(src_shape, indices_or_sections, axis):
    A = tvm.placeholder(shape=src_shape, name="A")
    tensor_l = topi.split(A, indices_or_sections, axis=axis)
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(tensor_l)
        ctx = tvm.context(device, 0)
        foo = tvm.build(s, [A] + tensor_l, device, name="split")
        data_npy = np.random.normal(size=src_shape).astype(A.dtype)
        out_npys = np.split(data_npy, indices_or_sections, axis=axis)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nds = [tvm.nd.empty(out_npy.shape, ctx=ctx, dtype=tensor_l[0].dtype) for out_npy in out_npys]
        foo(*([data_nd] + out_nds))
        for out_nd, out_npy in zip(out_nds, out_npys):
            np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    check_device("cuda")
    check_device("opencl")
    check_device("metal")
    check_device("rocm")

def test_expand_dims():
    verify_expand_dims((3, 10), (3, 10, 1, 1), 2, 2)
    verify_expand_dims((3, 10), (1, 3, 10), -3, 1)


def test_tranpose():
    verify_tranpose((3, 10, 2), (1, 0, 2))
    verify_tranpose((3, 10, 5), (2, 0, 1))
    verify_tranpose((3, 10), None)


def test_reshape():
    verify_reshape((1, 2, 3, 4), (2, 3, 4))
    verify_reshape((4, 2, 3, 4), (2, 4, 12))
    verify_reshape((4, 2, 3, 4), (2, 48))
    verify_reshape((16, ), (2, 2, 2, 2))


def test_squeeze():
    verify_squeeze((1, 2, 3, 4), 0)
    verify_squeeze((1, 2, 1, 4), None)
    verify_squeeze((1, 1, 1, 4), (1, 2))
    verify_squeeze((1, 1, 1, 1), None)


def test_concatenate():
    verify_concatenate([(2, 3, 4), (2, 2, 4), (2, 5, 4)], 1)
    verify_concatenate([(1, 2, 4), (1, 2, 3), (1, 2, 7), (1, 2, 8), (1, 2, 1)], -1)
    verify_concatenate([(5, 6, 7, 3),
                        (16, 6, 7, 3),
                        (12, 6, 7, 3),
                        (8, 6, 7, 3),
                        (2, 6, 7, 3)], 0)


def test_split():
    verify_split((2, 12, 3), 3, 1)
    verify_split((2, 12, 3), [2, 4], 1)
    verify_split((10, 12, 24), [5, 7, 9], -1)

if __name__ == "__main__":
    test_tranpose()
    test_expand_dims()
    test_reshape()
    test_squeeze()
    test_concatenate()
    test_split()
