"""Test code for broadcasting operators."""
import numpy as np
import tvm
import topi

def verify_expand_dims(in_shape, out_shape, axis, num_newaxis):
    A = tvm.placeholder(shape=in_shape, name="A")
    B = topi.cpp.expand_dims(A, axis, num_newaxis)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.schedule_injective(target, [B])
        else:
            s = topi.cpp.cuda.schedule_injective(target, [B])
        foo = tvm.build(s, [A, B], device, name="expand_dims")
        data_npy = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = data_npy.reshape(out_shape)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nd = tvm.nd.array(np.empty(out_shape).astype(B.dtype), ctx)
        foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in ["llvm", "nvptx", "cuda", "opencl", "metal", "rocm"]:
        check_device(device)


def verify_tranpose(in_shape, axes):
    A = tvm.placeholder(shape=in_shape, name="A")
    B = topi.cpp.transpose(A, axes)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.schedule_injective(target, [B])
        else:
            s = topi.cpp.cuda.schedule_injective(target, [B])
        ctx = tvm.context(device, 0)
        foo = tvm.build(s, [A, B], device, name="tranpose")
        data_npy = np.arange(np.prod(in_shape)).reshape(in_shape).astype(A.dtype)
        out_npy = data_npy.transpose(axes)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nd = tvm.nd.empty(out_npy.shape, ctx=ctx, dtype=B.dtype)
        foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in ["llvm", "nvptx", "cuda", "opencl", "metal", "rocm"]:
        check_device(device)


def verify_reshape(src_shape, dst_shape):
    A = tvm.placeholder(shape=src_shape, name="A")
    B = topi.cpp.reshape(A, dst_shape)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.schedule_injective(target, [B])
        else:
            s = topi.cpp.cuda.schedule_injective(target, [B])
        foo = tvm.build(s, [A, B], device, name="reshape")
        data_npy = np.random.normal(size=src_shape).astype(A.dtype)
        out_npy = np.reshape(data_npy, newshape=dst_shape)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nd = tvm.nd.empty(dst_shape, ctx=ctx, dtype=B.dtype)
        foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in ["llvm", "nvptx", "cuda", "opencl", "metal", "rocm"]:
        check_device(device)


def verify_squeeze(src_shape, axis):
    A = tvm.placeholder(shape=src_shape, name="A")
    B = topi.cpp.squeeze(A, axis)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.schedule_injective(target, [B])
        else:
            s = topi.cpp.cuda.schedule_injective(target, [B])
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

    for device in ["llvm", "nvptx", "cuda", "opencl", "metal", "rocm"]:
        check_device(device)

def verify_concatenate(shapes, axis):
    tensor_l = []
    for i, shape in enumerate(shapes):
        tensor_l.append(tvm.placeholder(shape, name="A" + str(i)))
    out_tensor = topi.cpp.concatenate(tensor_l, axis)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.schedule_injective(target, [out_tensor])
        else:
            s = topi.cpp.cuda.schedule_injective(target, [out_tensor])
        foo = tvm.build(s, tensor_l + [out_tensor], device, name="concatenate")
        data_npys = [np.random.normal(size=shape).astype(tensor_l[0].dtype) for shape in shapes]
        out_npy = np.concatenate(data_npys, axis=axis)
        data_nds = [tvm.nd.array(data_npy, ctx) for data_npy in data_npys]
        out_nd = tvm.nd.empty(out_npy.shape, ctx=ctx, dtype=out_tensor.dtype)
        foo(*(data_nds + [out_nd]))
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in ["llvm", "nvptx", "cuda", "opencl", "metal", "rocm"]:
        check_device(device)


def verify_split(src_shape, indices_or_sections, axis):
    A = tvm.placeholder(shape=src_shape, name="A")
    tensor_l = topi.cpp.split(A, indices_or_sections, axis)
    tensor_l = list(tensor_l)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.schedule_injective(target, tensor_l)
        else:
            s = topi.cpp.cuda.schedule_injective(target, tensor_l)
        ctx = tvm.context(device, 0)
        foo = tvm.build(s, [A] + tensor_l, device, name="split")
        data_npy = np.random.normal(size=src_shape).astype(A.dtype)
        out_npys = np.split(data_npy, indices_or_sections, axis=axis)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nds = [tvm.nd.empty(out_npy.shape, ctx=ctx, dtype=tensor_l[0].dtype) for out_npy in out_npys]
        foo(*([data_nd] + out_nds))
        for out_nd, out_npy in zip(out_nds, out_npys):
            np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in ["llvm", "nvptx", "cuda", "opencl", "metal", "rocm"]:
        check_device(device)

def verify_take(src_shape, indices_src, axis=None):
    src_dtype = "float32"
    indices_dtype = "int32"
    indices_src = np.array(indices_src, dtype=indices_dtype)
    A = tvm.placeholder(shape=src_shape, dtype=src_dtype, name="A")
    indices = tvm.placeholder(shape=indices_src.shape, dtype=indices_dtype, name="indices")
    if axis is None:
        out_tensor = topi.cpp.take(A, indices)
    else:
        out_tensor = topi.cpp.take(A, indices, axis)

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

    for device in ["llvm", "opencl"]:
        check_device(device)

def verify_where(condition, x, y):
    dtype = "float32"
    if len(condition.shape) == 1:
        np_out = np.array([xv if c else yv for (c,xv,yv) in zip(condition,x,y)])
    else:
        np_out = np.where(condition, x, y)
    A = tvm.placeholder(shape=condition.shape, dtype=dtype, name="condition")
    B = tvm.placeholder(shape=x.shape, dtype=dtype, name="x")
    C = tvm.placeholder(shape=y.shape, dtype=dtype, name="y")
    out_tensor = topi.cpp.where(A, B, C)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(out_tensor)

        foo = tvm.build(s, [A, B, C, out_tensor], device, name="where")
        tvm_out = tvm.nd.empty(x.shape, ctx=ctx, dtype=dtype)
        foo(tvm.nd.array(condition, ctx), tvm.nd.array(x, ctx),
            tvm.nd.array(y, ctx), tvm_out)
        np.testing.assert_allclose(tvm_out.asnumpy(), np_out)

    for device in ["llvm", "nvptx", "cuda", "opencl", "metal", "rocm"]:
        check_device(device)

def verify_concatenate_split(shapes, axis, indices_or_sections):
    tensor_l_concatenate = []
    for i, shape in enumerate(shapes):
        tensor_l_concatenate.append(tvm.placeholder(shape, name="A" + str(i)))
    out_tensor = topi.cpp.concatenate(tensor_l_concatenate, axis)
    tensor_l = topi.cpp.split(out_tensor, indices_or_sections, axis)
    tensor_l = list(tensor_l)
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.schedule_injective(target, tensor_l)
        else:
            s = topi.cpp.cuda.schedule_injective(target, tensor_l)
        ctx = tvm.context(device, 0)
        foo = tvm.build(s, tensor_l_concatenate + tensor_l, device, name="concatenate_split")
        data_npys = [np.random.normal(size=shape).astype(tensor_l_concatenate[0].dtype) for shape in shapes]
        out_npy_conc = np.concatenate(data_npys, axis=axis)
        out_npys_split = np.split(out_npy_conc, indices_or_sections, axis=axis)
        data_nds = [tvm.nd.array(data_npy, ctx) for data_npy in data_npys]
        out_nds = [tvm.nd.empty(out_npy.shape, ctx=ctx, dtype=tensor_l[0].dtype) for out_npy in out_npys_split]
        foo(*(data_nds + out_nds))
        for out_nd, out_npy in zip(out_nds, out_npys_split):
            np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for device in ["llvm", "cuda", "opencl", "metal", "rocm"]:
        check_device(device)

def verify_concatenate_broadcast(shapes, axis, rhs_shape):
    B = tvm.placeholder(shape=rhs_shape, name="B")
    tensor_l = []
    for i, shape in enumerate(shapes):
        tensor_l.append(tvm.placeholder(shape, name="A" + str(i)))
    out_tensor = topi.cpp.concatenate(tensor_l, axis)
    C = out_tensor + B
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.schedule_injective(target, [C])
        else:
            s = topi.cpp.cuda.schedule_injective(target, [C])
        ctx = tvm.context(device, 0)
        foo = tvm.build(s, tensor_l + [B, C], device, name="broadcast_binary_add")
        data_npys = [np.random.normal(size=shape).astype(tensor_l[0].dtype) for shape in shapes]
        lhs_npy = np.concatenate(data_npys, axis=axis)
        rhs_npy = np.random.uniform(size=rhs_shape).astype(B.dtype)
        out_npy = lhs_npy + rhs_npy
        data_nds = [tvm.nd.array(data_npy, ctx) for data_npy in data_npys]
        rhs_nd = tvm.nd.array(rhs_npy, ctx)
        out_nd = tvm.nd.array(np.empty(out_npy.shape).astype(B.dtype), ctx)
        for _ in range(1):
            foo(*(data_nds + [rhs_nd] + [out_nd]))
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy, rtol=1E-4, atol=1E-4)

    for device in ["llvm", "cuda", "opencl", "metal", "rocm"]:
        check_device(device)


def test_expand_dims():
    verify_expand_dims((3, 10), (3, 10, 1, 1), 2, 2)
    verify_expand_dims((3, 10), (1, 3, 10), -3, 1)


def test_tranpose():
    verify_tranpose((3, 10, 2), (1, 0, 2))
    verify_tranpose((3, 10, 5), (2, 0, 1))
    verify_tranpose((3, 10), None)
    verify_tranpose((3, 10, 5), (2, -3, 1))


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
    verify_split((2, 12, 3), 3, -1)
    verify_split((2, 12, 3), [2, 4], 1)
    verify_split((10, 12, 24), [5, 7, 9], -1)

def test_take():
    verify_take((4,), [1])
    verify_take((4,), [[0,1,2,3]])
    verify_take((3,3,3), [[11,25]])
    verify_take((4,), [[0,1],[2,3]])
    verify_take((4,), [1], 0)
    verify_take((2,2), [[[1,0],[0,1]]], 0)
    verify_take((2,2), [[[1,0],[0,1]]], 1)
    verify_take((4,3,5,6), [[2,1,0,0]], -2)

def test_where():
    shape = (10, 3, 7, 13)
    condition = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
    x = np.random.uniform(size=shape).astype("float32")
    y = np.random.uniform(size=shape).astype("float32")
    verify_where(condition, x, y)
    condition = np.random.uniform(low=-1, high=1, size=(shape[0],)).astype("float32")
    x = np.random.uniform(size=shape).astype("float32")
    y = np.random.uniform(size=shape).astype("float32")
    verify_where(condition, x, y)


def test_regression_1():
    verify_concatenate_split([(2, 3, 4), (2, 2, 4), (2, 5, 4)], 1, [3, 7])
    verify_concatenate_split([(3, 4), (2, 4), (3, 4)], 0, [1, 2, 3, 4])

def test_regression_2():
    verify_concatenate_broadcast([(5, 1, 3), (5, 1, 3)], 1, [2, 1])
    verify_concatenate_broadcast([(5, 1, 2), (5, 1, 3)], 2, [1, 5])

if __name__ == "__main__":
    test_concatenate()
    test_tranpose()
    test_expand_dims()
    test_reshape()
    test_squeeze()
    test_split()
    test_take()
    test_where()
    test_regression_1()
    test_regression_2()
