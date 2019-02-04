""" Support level10 operator test cases.
"""
import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing import ctx_list

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
        print(zz.checked_type)
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

if __name__ == "__main__":
    test_collapse_sum_like()
    test_broadcast_to_like()
    test_slice_like()
    test_reverse_reshape()
