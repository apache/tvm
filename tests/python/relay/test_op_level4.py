import tvm
import numpy as np
from tvm import relay


def test_binary_op():
    def check_binary_op(opfunc):
        n = tvm.var("n")
        t1 = relay.TensorType((5, n, 5))
        t2 = relay.TensorType((n, 1))
        x = relay.var("x", t1)
        y = relay.var("y", t2)
        z = opfunc(x, y)
        # test printer
        assert ("%0 = {}(%x, %y)".format(z.op.name)) in z.astext()
        assert relay.ir_pass.infer_type(z).checked_type == t1

    for opfunc in [relay.pow]:
        check_binary_op(opfunc)


def test_cmp_type():
    for op in (relay.greater,
               relay.greater_equal,
               relay.less,
               relay.less_equal,
               relay.equal,
               relay.not_equal):
        x = relay.var("x", relay.TensorType((10, 4), "float32"))
        y = relay.var("y", relay.TensorType((5, 10, 1), "float32"))
        z = op(x, y)
        z.astext()
        zz = relay.ir_pass.infer_type(z)
        assert zz.checked_type == relay.TensorType((5, 10, 4), "bool")


def test_binary_int_broadcast():
    for op in [relay.right_shift,
               relay.left_shift,
               relay.maximum,
               relay.minimum]:
        x = relay.var("x", relay.TensorType((10, 4), "int32"))
        y = relay.var("y", relay.TensorType((5, 10, 1), "int32"))
        z = op(x, y)
        zz = relay.ir_pass.infer_type(z)
        assert zz.checked_type == relay.TensorType((5, 10, 4), "int32")


def test_arg_reduce():
    for op in [relay.argmax, relay.argmin]:
        n, c , h, w = 10, 20, 3, 4
        x = relay.var("x", relay.ty.TensorType((n, c , h, w), "float32"))
        z = relay.argmax(x, axis=(1,))
        "axis="  in z.astext()
        zz = relay.ir_pass.infer_type(z)
        assert zz.checked_type == relay.ty.TensorType((n, h, w), "int32")
        n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
        x = relay.var("x", relay.ty.TensorType((n, c , h, w), "float32"))
        z = relay.argmax(x, axis=(2,), keepdims=True)
        zz = relay.ir_pass.infer_type(z)
        assert zz.checked_type == relay.ty.TensorType((n, c , 1, w), "int32")

        n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
        x = relay.var("x", relay.ty.TensorType((n, c , h, w), "float32"))
        z = relay.argmax(x, axis=(2,), keepdims=True, exclude=True)
        zz = relay.ir_pass.infer_type(z)
        assert zz.checked_type == relay.ty.TensorType((1, 1 , h, 1), "int32")
        ib = relay.ir_builder.IRBuilder()
        x = ib.param("x", relay.TensorType((10, 4), "int32"))
        y = ib.param("y", relay.TensorType((5, 10, 1), "int32"))
        with ib.function(x, y) as func:
            ib.ret(op(x, y))
        ib.ret(func)
        func = relay.ir_pass.infer_type(ib.env, func.to_func())
        ftype = func.checked_type
        assert ftype.ret_type == relay.TensorType((5, 10, 4), "int32")


def test_where():
    cond = relay.var("cond", relay.TensorType((3, 4), "float32"))
    x = relay.var("x", relay.TensorType((3, 4), "float32"))
    y = relay.var("y", relay.TensorType((3, 4), "float32"))
    z = relay.where(cond, x, y)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((3, 4), "float32")


def verify_reduce(test_func, data, axis, keepdims, exclude, output):
    ib = relay.ir_builder.IRBuilder()
    x = ib.param("x", relay.ty.TensorType(data, "float32"))
    with ib.function(x) as func:
        ib.ret(test_func(x, axis, keepdims, exclude))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    out_type = "int32" if test_func in [relay.argmin, relay.argmax] else "float32"
    assert ftype.ret_type == relay.ty.TensorType(output, out_type)

def test_reduce():
    d1, d2, d3, d4 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3"), tvm.var("d4")
    for func in [relay.sum,
                 relay.max,
                 relay.min,
                 relay.mean,
                 relay.prod,
                 relay.argmin,
                 relay.argmax]:
        verify_reduce(func, (d1, d2, d3, d4), (2,), True, False, (d1, d2, 1, d4))
        verify_reduce(func, (2, 3, 4), (1,), True, False, (2, 1, 4))
        verify_reduce(func, (2, 3, 4), (0, 1, 2), False, False, ())
        verify_reduce(func, (4, 4, 3), None, True, False, (1, 1, 1))
        verify_reduce(func, (4, 4, 3), None, False, False, ())
        verify_reduce(func, (4, 4, 3), (0, 2), False, False, (4,))
        verify_reduce(func, (128, 24, 128), (0, 1), False, False, (128,))
        verify_reduce(func, (128, 24, 128), (0, 2), False, False, (24,))
        verify_reduce(func, (128, 24, 128), (0, 1), True, False, (1, 1, 128))
        verify_reduce(func, (128, 24, 128), (0, 2), True, False, (1, 24, 1))
        verify_reduce(func, (128, 24, 128), None, True, False, (1, 1, 1))

if __name__ == "__main__":
    test_binary_op()
    test_cmp_type()
    test_binary_int_broadcast()
    test_where()
    test_arg_reduce()
    test_multibox_prior()
    test_reduce()
