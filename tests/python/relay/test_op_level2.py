""" Support level2 operator test cases.
"""
import tvm
from tvm import relay


def test_conv2d_infer_type():
    # symbolic in batch dimension
    n, c, h, w = tvm.var("n"), 10, 224, 224
    x = relay.var("x", relay.ty.TensorType((n, c, h, w), "float32"))
    w = relay.var("w")
    y = relay.nn.conv2d(x, w,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        channels=2)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type ==  relay.TensorType(
        (n, 2, 224, 224), "float32")
    assert yy.args[1].checked_type == relay.TensorType(
        (2, 10, 3, 3), "float32")

    # infer by shape of w, mixed precision

    n, c, h, w = tvm.var("n"), 10, 224, 224
    x = relay.var("x", relay.TensorType((n, c, h, w), "int8"))
    w = relay.var("w", relay.TensorType((2, 10, 3, 3), "int8"))
    y = relay.nn.conv2d(x, w, out_dtype="int32")
    assert "out_dtype=\"int32\"" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type ==  relay.TensorType(
        (n, 2, 222, 222), "int32")

    # Infer with a different layout
    n, c, h, w = 4, 32, 224, 224
    x = relay.var("x", relay.TensorType((n//4, c//4, h, w, 4, 4), "int8"))
    wt = relay.var("w")
    y = relay.nn.conv2d(x, wt,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        channels=16,
                        data_layout="NCHW4n4c",
                        weight_layout="OIHW4o4i",
                        out_dtype="int32")
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type ==  relay.TensorType(
        (1, 4, 224, 224, 4, 4), "int32")
    assert yy.args[1].checked_type == relay.TensorType(
        (4, 8, 3, 3, 4, 4), "int8")

    # Infer with NHWC
    n, c, h, w = 4, 32, 224, 224
    x = relay.var("x", relay.TensorType((n, h, w, c), "int8"))
    wt = relay.var("w")
    y = relay.nn.conv2d(x, wt,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        channels=16,
                        data_layout="NHWC",
                        out_dtype="int32")
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type ==  relay.TensorType(
        (n, h, w, 16), "int32")


def test_conv2d_transpose_infer_type():
    # symbolic in batch dimension
    n, c, h, w = tvm.var("n"), 10, 10, 12
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    w = relay.var("w", relay.IncompleteType())
    y = relay.nn.conv2d_transpose(x, w,
                                  kernel_size=(3, 3),
                                  padding=(1, 1),
                                  channels=15)
    assert "channels=15" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (n, 15, 10, 12), "float32")
    assert yy.args[1].checked_type == relay.TensorType(
        (10, 15, 3, 3), "float32")

    # infer by shape of w, mixed precision
    n, c, h, w = tvm.var("n"), 10, 10, 12
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    w = relay.var("w", relay.TensorType((12, 11, 5, 5), "float32"))
    y = relay.nn.conv2d_transpose(x, w,
                                  output_padding=(1, 1),
                                  channels=11,
                                  data_layout="NHWC")
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (n, 15, 15, 11), "float32")

def test_upsampling_infer_type():
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = relay.nn.upsampling(x, scale=2, layout="NCHW", method="BILINEAR")
    "method=\"BINLINEAR\"" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h*2, w*2), "float32")
    n, c = tvm.var("n"), tvm.var("c")
    x = relay.var("x", relay.TensorType((n, c, 100, 200), "float32"))
    y = relay.nn.upsampling(x, scale=2, layout="NCHW", method="BILINEAR")
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, 200, 400), "float32")

def _test_pool2d_infer_type(opfunc):
    n, c, h, w = tvm.var("n"), 10, 224, 224
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = opfunc(x, pool_size=(1, 1))
    assert "pool_size=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 10, 224, 224), "float32")

def _test_global_pool2d_infer_type(opfunc):
    n, c, h, w = tvm.var("n"), tvm.var("c"), 224, 224
    x = relay.var("x", relay.TensorType((n, h, w, c), "float32"))
    y = opfunc(x, layout="NHWC")
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 1, 1, c), "float32")

    n, c, h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = opfunc(x)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, 1, 1), "float32")

def test_pool2d_infer_type():
    _test_pool2d_infer_type(relay.nn.max_pool2d)
    _test_pool2d_infer_type(relay.nn.avg_pool2d)
    _test_global_pool2d_infer_type(relay.nn.global_avg_pool2d)
    _test_global_pool2d_infer_type(relay.nn.global_avg_pool2d)

def test_flatten_infer_type():
    d1, d2, d3, d4 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3"), tvm.var("d4")
    x = relay.var("x", relay.TensorType((d1, d2, d3, d4), "float32"))
    y = relay.nn.batch_flatten(x)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((d1, ((d2*d3)*d4)), "float32")

    x = relay.var("x", relay.TensorType((3, 2, 4, 3), "float32"))
    y = relay.nn.batch_flatten(x)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((3, 24), "float32")

    x = relay.var("x", relay.TensorType((d1, 2, d3, 3), "float32"))
    y = relay.nn.batch_flatten(x)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((d1, ((2*d3)*3)), "float32")

def test_pad_infer_type():
    # entirely concrete case
    n, c, h, w = 1, 2, 3, 4
    t = relay.var("t", relay.TensorType((n, c, h, w), "float32"))
    y = relay.nn.pad(t, ((1, 1), (2, 2), (3, 3), (4, 4)))
    "pad_width=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((3, 6, 9, 12), "float32")

    # some symbolic values
    n, c, h, w = tvm.var("n"), 2, 3, tvm.var("w")
    t = relay.var("t", relay.TensorType((n, c, h, w), "float32"))
    y = relay.nn.pad(t, ((1, 1), (2, 2), (3, 3), (4, 4)))
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n + 2, 6, 9, w + 8), "float32")

def test_dense_infer_type():
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    w = relay.var("w", relay.TensorType((w, 2), "float32"))
    y = relay.nn.dense(x, w, units=2)
    "units=2" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, 2), "float32")

    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), 2
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    wh, ww = tvm.var("wh"), tvm.var("ww")
    w = relay.var("w", relay.TensorType((wh, ww), "float32"))
    y = relay.nn.dense(x, w)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, ww), "float32")

    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), 2
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    w = relay.var("w", relay.IncompleteType())
    y = relay.nn.dense(x, w, units=2)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, 2), "float32")


def test_lrn():
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = relay.var("x", shape=(n, c , h, w))
    y = relay.nn.lrn(x, size=10, axis=2, bias=0.5, alpha=.00001, beta=0.75)
    "alpha=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c , h, w))

def test_l2_normalize():
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = relay.var("x", shape=(n, c , h, w))
    y = relay.nn.l2_normalize(x, eps=0.001, axis=[1])
    "axis=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c , h, w))


if __name__ == "__main__":
    test_lrn()
    test_l2_normalize()
    test_conv2d_infer_type()
    test_pool2d_infer_type()
    test_upsampling_infer_type()
    test_flatten_infer_type()
    test_pad_infer_type()
    test_conv2d_transpose_infer_type()
    test_dense_infer_type()
