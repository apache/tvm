"""Test code for broadcasting operators."""
import numpy as np
import tvm
import topi

from common import get_all_backend

def verify_expand_dims(in_shape, out_shape, axis, num_newaxis):
    A = tvm.placeholder(shape=in_shape, name="A")
    B = topi.expand_dims(A, axis, num_newaxis)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_broadcast(B)
        foo = tvm.build(s, [A, B], device, name="expand_dims")
        data_npy = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = data_npy.reshape(out_shape)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nd = tvm.nd.array(np.empty(out_shape).astype(B.dtype), ctx)
        foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in get_all_backend():
        check_device(device)


def verify_tranpose(in_shape, axes):
    A = tvm.placeholder(shape=in_shape, name="A")
    B = topi.transpose(A, axes)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(B)
        foo = tvm.build(s, [A, B], device, name="tranpose")
        data_npy = np.arange(np.prod(in_shape)).reshape(in_shape).astype(A.dtype)
        out_npy = data_npy.transpose(axes)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nd = tvm.nd.empty(out_npy.shape, ctx=ctx, dtype=B.dtype)
        foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in get_all_backend():
        check_device(device)


def verify_reshape(src_shape, dst_shape):
    A = tvm.placeholder(shape=src_shape, name="A")
    B = topi.reshape(A, dst_shape)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(B)
        foo = tvm.build(s, [A, B], device, name="reshape")
        data_npy = np.random.normal(size=src_shape).astype(A.dtype)
        out_npy = np.reshape(data_npy, newshape=dst_shape)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nd = tvm.nd.empty(dst_shape, ctx=ctx, dtype=B.dtype)
        foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in get_all_backend():
        check_device(device)


def verify_squeeze(src_shape, axis):
    A = tvm.placeholder(shape=src_shape, name="A")
    B = topi.squeeze(A, axis=axis)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(B)

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

    for device in get_all_backend():
        check_device(device)

def verify_concatenate(shapes, axis):
    tensor_l = []
    for i, shape in enumerate(shapes):
        tensor_l.append(tvm.placeholder(shape, name="A" + str(i)))
    out_tensor = topi.concatenate(a_tuple=tensor_l, axis=axis)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(out_tensor)

        foo = tvm.build(s, tensor_l + [out_tensor], device, name="concatenate")
        data_npys = [np.random.normal(size=shape).astype(tensor_l[0].dtype) for shape in shapes]
        out_npy = np.concatenate(data_npys, axis=axis)
        data_nds = [tvm.nd.array(data_npy, ctx) for data_npy in data_npys]
        out_nd = tvm.nd.empty(out_npy.shape, ctx=ctx, dtype=out_tensor.dtype)
        foo(*(data_nds + [out_nd]))
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in get_all_backend():
        check_device(device)


def verify_split(src_shape, indices_or_sections, axis):
    A = tvm.placeholder(shape=src_shape, name="A")
    tensor_l = topi.split(A, indices_or_sections, axis=axis)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(tensor_l)

        foo = tvm.build(s, [A] + tensor_l, device, name="split")
        data_npy = np.random.normal(size=src_shape).astype(A.dtype)
        out_npys = np.split(data_npy, indices_or_sections, axis=axis)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nds = [tvm.nd.empty(out_npy.shape, ctx=ctx, dtype=tensor_l[0].dtype) for out_npy in out_npys]
        foo(*([data_nd] + out_nds))
        for out_nd, out_npy in zip(out_nds, out_npys):
            np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in get_all_backend():
        check_device(device)


def verify_expand_like(in_shape, out_shape, axis):
    A = tvm.placeholder(shape=in_shape, name="A")
    B = tvm.placeholder(shape=out_shape, name="B")
    C = topi.expand_like(A, B, axis)
    s = tvm.create_schedule([C.op])

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        ctx = tvm.context(device, 0)
        f = tvm.build(s, [A, B, C], device, name="expand_like")
        input = np.random.uniform(size=in_shape).astype(A.dtype)
        tvm_input = tvm.nd.array(input, ctx)

        odim = len(out_shape)
        real_axis = [x if x >= 0 else x + odim for x in axis]
        real_axis = sorted(real_axis)
        for x in real_axis:
            input = np.expand_dims(input, x).astype(A.dtype)
        for x in real_axis:
            input = np.concatenate([input]*out_shape[x], axis=x).astype(A.dtype)
        assert input.shape == out_shape

        tvm_shape_like = tvm.nd.array(np.zeros(out_shape).astype(B.dtype), ctx)
        out = tvm.nd.array(np.zeros(out_shape).astype(A.dtype), ctx)
        f(tvm_input, tvm_shape_like, out)
        np.testing.assert_allclose(out.asnumpy(), input)

    for device in ["llvm"]:
        check_device(device)

def verify_flip(in_shape, axis):
    A = tvm.placeholder(shape=in_shape, name="A")
    B = topi.flip(A, axis) + 1
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(B)

        foo = tvm.build(s, [A, B], device, name="reverse")
        x_np = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = np.flip(x_np, axis) + 1
        data_nd = tvm.nd.array(x_np, ctx)
        out_nd = tvm.nd.empty(out_npy.shape, ctx=ctx, dtype=A.dtype)
        foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in ["llvm", "cuda", "opencl", "sdaccel", "aocl_sw_emu"]:
        check_device(device)

def verify_take(src_shape, indices_src, axis=None):
    src_dtype = "float32"
    indices_dtype = "int32"
    indices_src = np.array(indices_src, dtype=indices_dtype)
    A = tvm.placeholder(shape=src_shape, dtype=src_dtype, name="A")
    indices = tvm.placeholder(shape=indices_src.shape, dtype=indices_dtype, name="indices")
    if axis is None:
        out_tensor = topi.take(a=A, indices=indices)
    else:
        out_tensor = topi.take(a=A, indices=indices, axis=axis)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(out_tensor)

        foo = tvm.build(s, [A] + [indices] + [out_tensor] , device, name="take")
        shape_size = 1
        for i in range(len(src_shape)):
            shape_size = shape_size * src_shape[i]
        data_npy = np.arange(shape_size, dtype=src_dtype).reshape((src_shape))

        if axis is None:
            out_npys = np.take(data_npy, indices_src)
        else:
            out_npys = np.take(data_npy, indices_src, axis=axis)
        data_nd = tvm.nd.array(data_npy, ctx)
        indices_nd = tvm.nd.array(indices_src, ctx)
        out_nd = tvm.nd.empty(out_npys.shape, ctx=ctx, dtype=src_dtype)
        foo(data_nd, indices_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npys)

    for device in ["llvm", "opencl", "sdaccel", "aocl_sw_emu"]:
        check_device(device)

def verify_strided_slice(in_shape, begin, end, stride=None):
    stride = stride if stride else [1, 1, 1]
    A = tvm.placeholder(shape=in_shape, name="A")
    B = topi.strided_slice(A, begin, end, stride) + 1
    def test_forward(x, begin, end, stride):
        return x[begin[0]:end[0]:stride[0],
                    begin[1]:end[1]:stride[1], begin[2]:end[2]:stride[2]] + 1
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(B)

        foo = tvm.build(s, [A, B], device, name="stride_slice")
        x_np = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = test_forward(x_np, begin, end, stride)
        data_nd = tvm.nd.array(x_np, ctx)
        out_nd = tvm.nd.empty(out_npy.shape, ctx=ctx, dtype=A.dtype)
        foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in ["llvm", "opencl", "sdaccel", "aocl_sw_emu"]:
        check_device(device)

def test_strided_slice():
    verify_strided_slice((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2])
    verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1])
    verify_strided_slice((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1])
    verify_strided_slice((3, 4, 3), [1, 0, 0], [2, 2, 3], [1, 1, 2])
    verify_strided_slice((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1])
    verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 4, 3])

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

    # a special case to trigger inline let expression
    A = tvm.placeholder((2,), 'float32', 'A')
    E = topi.squeeze(A)
    C = tvm.compute((1,), lambda i: E[(2 * A[0] - 1).astype('int32')])
    for device in ['cuda', 'opencl']:
        ctx = tvm.context(device, 0)
        if ctx.exist:
            with tvm.target.create(device):
                s = topi.generic.schedule_injective(C)
                func = tvm.build(s, [A, C])
            a = tvm.nd.array(np.array((1, 2)).astype('float32'), ctx=ctx)
            c = tvm.nd.empty((1,), dtype='float32', ctx=ctx)
            func(a, c)
            assert c.asnumpy()[0] == 2


def test_concatenate():
    verify_concatenate([(2,), (2,), (2,)], 0)
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

def test_flip():
    verify_flip((3, 4, 3), 1)
    verify_flip((3, 4, 3), 0)
    verify_flip((3, 4, 3), 2)
    verify_flip((3, 4, 3), -1)
    verify_flip((3, 4, 3), -3)
    verify_flip((3, 4, 3), -2)

def test_expand_like():
    verify_expand_like((3,), (2, 3), [0])
    verify_expand_like((2,), (2, 3), [1])
    verify_expand_like((3, 4), (3, 5, 4), [1])
    verify_expand_like((5, 7), (5, 6, 7, 8), [1, 3])

def test_take():
    verify_take((4,), [1])
    verify_take((4,), [[0,1,2,3]])
    verify_take((3,3,3), [[11,25]])
    verify_take((4,), [[0,1],[2,3]])
    verify_take((4,), [1], 0)
    verify_take((2,2), [[[1,0],[0,1]]], 0)
    verify_take((2,2), [[[1,0],[0,1]]], 1)
    verify_take((4,3,5,6), [[2,1,0,0]], -2)

if __name__ == "__main__":
    test_concatenate()
    test_tranpose()
    test_expand_dims()
    test_reshape()
    test_squeeze()
    test_split()
    test_flip()
    test_expand_like()
    test_take()
    test_strided_slice()
