import tvm
import numpy as np
from tvm import relay
from tvm.relay.testing import ctx_list
import topi.testing

def test_binary_op():
    def check_binary_op(opfunc, ref):
        n = tvm.var("n")
        t1 = relay.TensorType((5, n, 5))
        t2 = relay.TensorType((n, 1))
        x = relay.var("x", t1)
        y = relay.var("y", t2)
        z = opfunc(x, y)
        # test printer
        assert ("%0 = {}(%x, %y)".format(z.op.name)) in z.astext()
        assert relay.ir_pass.infer_type(z).checked_type == t1

        if ref is not None:
            t1 = relay.TensorType((5, 10, 5))
            t2 = relay.TensorType((5, 10, 5))
            x = relay.var("x", t1)
            y = relay.var("y", t2)
            z = opfunc(x, y)
            x_data = np.random.rand(5, 10, 5).astype(t1.dtype)
            y_data = np.random.rand(5, 10, 5).astype(t2.dtype)
            ref_res = ref(x_data, y_data)
            func = relay.Function([x, y], z)

            for target, ctx in ctx_list():
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)

    for opfunc, ref in [(relay.power, np.power)]:
        check_binary_op(opfunc, ref)


def test_cmp_type():
    for op, ref in ((relay.greater, np.greater),
               (relay.greater_equal, np.greater_equal),
               (relay.less, np.less),
               (relay.less_equal, np.less_equal),
               (relay.equal, np.equal),
               (relay.not_equal, np.not_equal)):
        x = relay.var("x", relay.TensorType((10, 4), "float32"))
        y = relay.var("y", relay.TensorType((5, 10, 1), "float32"))
        z = op(x, y)
        z.astext()
        zz = relay.ir_pass.infer_type(z)
        assert zz.checked_type == relay.TensorType((5, 10, 4), "bool")

        if ref is not None:
            x_shape = (10, 4)
            y_shape = (5, 10, 1)
            t1 = relay.TensorType(x_shape)
            t2 = relay.TensorType(y_shape)
            x = relay.var("x", t1)
            y = relay.var("y", t2)
            z = op(x, y)
            x_data = np.random.rand(*x_shape).astype(t1.dtype)
            y_data = np.random.rand(*y_shape).astype(t2.dtype)
            ref_res = ref(x_data, y_data)
            func = relay.Function([x, y], z)

            for target, ctx in ctx_list():
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)


def test_binary_int_broadcast():
    for op, ref in [(relay.right_shift, np.right_shift),
               (relay.left_shift, np.left_shift),
                (relay.mod, np.mod),
               (relay.maximum, np.maximum),
               (relay.minimum, np.minimum)]:
        x = relay.var("x", relay.TensorType((10, 4), "int32"))
        y = relay.var("y", relay.TensorType((5, 10, 1), "int32"))
        z = op(x, y)
        zz = relay.ir_pass.infer_type(z)
        assert zz.checked_type == relay.TensorType((5, 10, 4), "int32")

    if ref is not None:
        x_shape = (10, 4)
        y_shape = (5, 10, 1)
        t1 = relay.TensorType(x_shape, 'int32')
        t2 = relay.TensorType(y_shape, 'int32')
        x_data = np.random.rand(*x_shape).astype(t1.dtype)
        y_data = np.random.rand(*y_shape).astype(t2.dtype)
        func = relay.Function([x, y], z)
        ref_res = ref(x_data, y_data)

        for target, ctx in ctx_list():
            intrp = relay.create_executor("graph", ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x_data, y_data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)


def test_where():
    cond = relay.var("cond", relay.TensorType((3, 4), "float32"))
    x = relay.var("x", relay.TensorType((3, 4), "float32"))
    y = relay.var("y", relay.TensorType((3, 4), "float32"))
    z = relay.where(cond, x, y)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((3, 4), "float32")


def verify_reduce(test_func, data, axis, keepdims, exclude, output):
    x = relay.var("x", relay.TensorType(data, "float32"))
    z = test_func(x, axis, keepdims, exclude)
    zz = relay.ir_pass.infer_type(z)
    if axis:
        assert "axis=" in z.astext()
    if keepdims:
        assert "keepdims=" in z.astext()
    if exclude:
        assert "exclude=" in z.astext()
    out_type = "int32" if test_func in [relay.argmin, relay.argmax] else "float32"
    assert zz.checked_type == relay.ty.TensorType(output, out_type)

def test_reduce_functions():
    d1, d2, d3, d4 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3"), tvm.var("d4")
    for func in [relay.sum,
                 relay.max,
                 relay.min,
                 relay.mean,
                 relay.prod,
                 relay.argmin,
                 relay.argmax]:
        verify_reduce(func, (d1, d2, d3, d4), (2,), True, False, (d1, d2, 1, d4))
        verify_reduce(func, (d1, d2, d3), (1,), True, False, (d1, 1, d3))
        verify_reduce(func, (d1, d2, d3), None, True, False, (1, 1, 1))
        verify_reduce(func, (d1, d2, d3), (0, 1), True, False, (1, 1, d3))
        verify_reduce(func, (2, 3, 4), (1,), True, False, (2, 1, 4))
        verify_reduce(func, (2, 3, 4), (0, 1, 2), False, False, ())
        verify_reduce(func, (4, 4, 3), None, True, False, (1, 1, 1))
        verify_reduce(func, (4, 4, 3), None, False, True, ())
        verify_reduce(func, (4, 4, 3), (0, 2), False, False, (4,))
        verify_reduce(func, (128, 24, 128), (0, 1), False, False, (128,))
        verify_reduce(func, (128, 24, 128), (0, 2), False, False, (24,))
        verify_reduce(func, (128, 24, 128), (0, 1), True, False, (1, 1, 128))
        verify_reduce(func, (128, 24, 128), (0, 2), True, False, (1, 24, 1))


def test_strided_slice():
    def verify(dshape, begin, end, strides, output, test_ref=True):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.strided_slice(x, begin=begin, end=end, strides=strides)
        func = relay.Function([x], z)
        func = relay.ir_pass.infer_type(func)
        text = func.astext()
        assert "begin=" in text
        assert "end=" in text
        if output:
            assert func.body.checked_type == relay.ty.TensorType(output, "float32")
        if not test_ref:
            return
        x_data = np.random.uniform(size=dshape).astype("float32")
        ref_res = topi.testing.strided_slice_python(
            x_data, begin, end, strides)
        for target, ctx in ctx_list():
            intrp = relay.create_executor("graph", ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x_data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)

    d1, d2, d3, d4 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3"), tvm.var("d4")
    verify((d1, d2, 3), [None, None, 1], [None, None, 2], None, (d1, d2, 1), False)
    verify((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2], (3, 1, 2))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], (1, 3, 3))
    verify((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], (1, 4, 3))
    verify((3, 4, 3), [1, 0, 0], [2, 2, 3], [1, 1, 2], (1, 2, 2))
    verify((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1], (1, 2, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 1000, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 4], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1], [4, 4, 3], None, (2, 3, 3))


if __name__ == "__main__":
    test_strided_slice()
    test_binary_op()
    test_cmp_type()
    test_binary_int_broadcast()
    test_where()
    test_reduce_functions()
