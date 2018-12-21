""" Support level10 operator test cases.
"""
import numpy as np
import tvm
import topi.testing
from tvm import relay
from tvm.relay.testing import ctx_list
import topi
import topi.testing

def test_collapse_sum_like():
    shape = (3, 4, 5, 6)
    shape_like = (4, 5, 6)
    dtype = "float32"
    x = relay.Var("x", relay.ty.TensorType(shape , dtype))
    y = relay.Var("y", relay.ty.TensorType(shape_like, dtype))
    z = relay.collapse_sum_like(x, y)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.ty.TensorType(shape_like, dtype)

    func = relay.Function([x, y], z)
    x = np.random.uniform(size=shape).astype(dtype)
    y = np.random.uniform(size=shape_like).astype(dtype)
    ref_res = np.sum(x, 0)
    for target, ctx in ctx_list():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x, y)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

def test_broadcast_to():
    shape = (4, 1, 6)
    shape_like = (3, 4, 5, 6)
    dtype = "float32"
    x = relay.Var("x", relay.ty.TensorType(shape , dtype))
    z = relay.broadcast_to(x, shape=shape_like)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.ty.TensorType(shape_like, dtype)

    func = relay.Function([x], z)
    x = np.random.uniform(size=shape).astype(dtype)
    ref_res = np.broadcast_to(x, shape_like)
    for target, ctx in ctx_list():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

def test_broadcast_to_like():
    shape = (4, 1, 6)
    shape_like = (3, 4, 5, 6)
    dtype = "float32"
    x = relay.Var("x", relay.ty.TensorType(shape , dtype))
    y = relay.Var("y", relay.ty.TensorType(shape_like, dtype))
    z = relay.broadcast_to_like(x, y)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.ty.TensorType(shape_like, dtype)

    func = relay.Function([x, y], z)
    x = np.random.uniform(size=shape).astype(dtype)
    y = np.random.uniform(size=shape_like).astype(dtype)
    ref_res = np.broadcast_to(x, shape_like)
    for target, ctx in ctx_list():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x, y)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)


def np_slice_like(np_data, np_shape_like, axis=None):
    begin_idx = [0 for _ in np_data.shape]
    end_idx = list(np_data.shape)
    if axis:
        for i in axis:
            if i < 0:
                i = len(np_data.shape) + i
            end_idx[i] = np_shape_like.shape[i]
    else:
        for i in range(len(np_data.shape)):
            if i < len(np_shape_like.shape):
                end_idx[i] = np_shape_like.shape[i]
    slice_idx = []
    for b, e in zip(begin_idx, end_idx):
        slice_idx.append(slice(b, e))
    np_result = np_data[tuple(slice_idx)]
    return np_result


def verify_slice_like(data, slice_like, axes, output, dtype="float32"):
    x = relay.var("data", relay.TensorType(data, dtype))
    y = relay.var("slice_like", relay.TensorType(slice_like, dtype))
    z = relay.slice_like(x, y, axes)
    zz = relay.ir_pass.infer_type(z)
    if axes:
        assert "axes" in z.astext()
    assert zz.checked_type == relay.ty.TensorType(output, dtype)

    if all(isinstance(v, int) == 0 for v in data) or \
        all(isinstance(v, int) == 0 for v in slice_like):
        return

    func = relay.Function([x, y], z)
    x_data = np.random.uniform(size=data).astype(dtype)
    y_data = np.random.uniform(size=slice_like).astype(dtype)
    ref_res = np_slice_like(x_data, y_data, axes)

    for target, ctx in ctx_list():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x_data, y_data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

def test_slice_like():
    d1, d2, d3, d4 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3"), tvm.var("d4")
    verify_slice_like(data=(d1, d2, d3), slice_like=(1, 2, 3), axes=None, output=(1, 2, 3))
    verify_slice_like(data=(1, 2, 3), slice_like=(d1, d2, d3), axes=None, output=(d1, d2, d3))
    verify_slice_like(data=(d2, d3, d4), slice_like=(d1, d2, d3), axes=(1,2), output=(d2, d2, d3))
    verify_slice_like(data=(3, 4, 5), slice_like=(1, 2, 3), axes=None, output=(1, 2, 3))
    verify_slice_like(data=(3, 4, 5), slice_like=(1, 2), axes=None, output=(1, 2, 5))
    verify_slice_like(data=(3, 4, 5), slice_like=(1, 2, 3), axes=(1, 2), output=(3, 2, 3))
    verify_slice_like(data=(3, 4, 5), slice_like=(1, 2, 3), axes=(-1, -3), output=(1, 4, 3))
    verify_slice_like(data=(1, 3, 224, 224),
                      slice_like=(1, 3, 112, 112),
                      axes=(2, 3),
                      output=(1, 3, 112, 112))

def test_reverse_reshape():
    def verify_reverse_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.reverse_reshape(x, newshape=newshape)
        zz = relay.ir_pass.infer_type(z)
        assert "newshape=" in z.astext()
        assert zz.checked_type == relay.ty.TensorType(oshape, "float32")

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        ref_res = np.reshape(x_data, oshape)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
    verify_reverse_reshape((2, 3, 4), (4, 0, 2), (4, 3, 2))
    verify_reverse_reshape((2, 3, 4), (2, 0, 0), (2, 3, 4))
    verify_reverse_reshape((2, 3, 4), (0, -1), (3, 8))
    verify_reverse_reshape((2, 3, 4), (-1, 0), (6, 4))
    verify_reverse_reshape((2, 3, 4), (0, -3), (2, 12))

def verify_batch_matmul(x_shape, y_shape, out_shape, dtype="float32"):
    x = relay.var("x", relay.TensorType(x_shape, dtype))
    y = relay.var("y", relay.TensorType(y_shape, dtype))
    z = relay.nn.batch_matmul(x, y)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.ty.TensorType(out_shape, dtype)

    func = relay.Function([x, y], z)
    x_np = np.random.uniform(size=x_shape).astype(dtype)
    y_np = np.random.uniform(size=y_shape).astype(dtype)
    z_np = topi.testing.batch_matmul(x_np, y_np)

    for target, ctx in ctx_list():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, ctx=ctx, target=target)
            z = intrp.evaluate(func)(x_np, y_np)
            tvm.testing.assert_allclose(z.asnumpy(), z_np, rtol=1e-5)

def test_batch_matmul():
    b, m, n, k = tvm.var("b"), tvm.var("m"), tvm.var("n"), tvm.var("k")
    x = relay.var("x", relay.TensorType((b, m, k), "float32"))
    y = relay.var("y", relay.TensorType((b, n, k), "float32"))
    z = relay.nn.batch_matmul(x, y)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((b, m, n), "float32")

    verify_batch_matmul((1, 16, 32), (1, 16, 32), (1, 16, 16))
    verify_batch_matmul((5, 16, 32), (5, 16, 32), (5, 16, 16))
    verify_batch_matmul((5, 16, 32), (5, 20, 32), (5, 16, 20))
    verify_batch_matmul((30, 16, 32), (30, 20, 32), (30, 16, 20))

def test_shape_of():
    shape = (10, 5, 12)
    x = relay.var("x", shape=shape)
    func = relay.Function([x], relay.op.shape_of(x))
    func = relay.ir_pass.infer_type(func)
    x_data = np.random.rand(*shape).astype('float32')
    for target, ctx in ctx_list():
        # Because using graph executor, this op will be optimized after
        # constant folding pass, here we only test with interpreter
        for kind in ["debug"]:
            intrp = relay.create_executor(kind, ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x_data)
            tvm.testing.assert_allclose(op_res.asnumpy(),
                                        np.array(shape).astype('int32'))

def test_deformable_conv2d():
    def test_infer_type(batch, in_channel, size, out_channel, deformable_groups, groups):
        data_shape = (batch, in_channel, size, size)
        data = relay.var("data", shape=data_shape)
        offset = relay.var("offset")
        kernel = relay.var("kernel")
        kernel_size = (3, 3)
        y = relay.nn.deformable_conv2d(data, offset, kernel,
            strides=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            kernel_size=kernel_size,
            deformable_groups=deformable_groups,
            groups=groups,
            channels=out_channel)
        weight_shape = (out_channel, in_channel // groups, kernel_size[0], kernel_size[1])
        out_shape = (batch, out_channel, size, size)
        offset_shape = (batch, 2 * kernel_size[0] * kernel_size[1] * deformable_groups, out_shape[2], out_shape[3])
        yy = relay.ir_pass.infer_type(y)
        assert yy.checked_type == relay.TensorType(out_shape)
        assert yy.args[1].checked_type == relay.TensorType(offset_shape), yy.args[1].checked_type
        assert yy.args[2].checked_type == relay.TensorType(weight_shape)

    test_infer_type(1, 4, 16, 4, 4, 1)
    test_infer_type(2, 4, 16, 4, 1, 2)


    def test_run(batch, in_channel, size, out_channel, deformable_groups, groups):
        kernel_size = (3, 3)
        data_shape = (batch, in_channel, size, size)
        offset_shape = (batch, 2 * kernel_size[0] * kernel_size[1] * deformable_groups, size, size)
        kernel_shape = (out_channel, in_channel // groups, kernel_size[0], kernel_size[1])
        dtype = 'float32'
        data = relay.var("data", shape=data_shape, dtype=dtype)
        offset = relay.var("offset")
        kernel = relay.var("kernel")
        y = relay.nn.deformable_conv2d(data, offset, kernel,
            strides=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            kernel_size=kernel_size,
            deformable_groups=deformable_groups,
            groups=groups,
            channels=out_channel)
        func = relay.Function([data, offset, kernel], y)
        data = np.random.uniform(size=data_shape).astype(dtype)
        offset = np.random.uniform(size=offset_shape).astype(dtype)
        kernel = np.random.uniform(size=kernel_shape).astype(dtype)
        ref_res = topi.testing.deformable_conv2d_nchw_python(data, offset, kernel, stride=(1, 1), padding=(1, 1), dilation=(1, 1), deformable_groups=deformable_groups, groups=groups)

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp1 = relay.create_executor(kind, ctx=ctx, target=target)
                op_res1 = intrp1.evaluate(func)(data, offset, kernel)
                tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5, atol=1e-5)
    test_run(1, 4, 16, 4, 1, 1)
    test_run(2, 4, 16, 4, 4, 1)


if __name__ == "__main__":
    test_collapse_sum_like()
    test_broadcast_to_like()
    test_slice_like()
    test_reverse_reshape()
    test_batch_matmul()
    test_shape_of()
    test_deformable_conv2d()
