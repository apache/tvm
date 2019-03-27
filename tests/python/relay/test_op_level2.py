""" Support level2 operator test cases.
"""
import tvm
from tvm import relay
from tvm.relay.testing import ctx_list
import numpy as np
import topi.testing

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
                        kernel_layout="OIHW4o4i",
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


def test_conv2d_run():
    def run_test_conv2d(dtype, out_dtype, scale, dshape, kshape,
                        padding=(1, 1),
                        fref=None,
                        groups=1,
                        dilation=(1, 1),
                        **attrs):
        x = relay.var("x", shape=dshape)
        w = relay.var("w")
        y = relay.nn.conv2d(x, w,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            **attrs)
        func = relay.Function([x, w], y)
        data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
        kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype)
        dkernel = topi.testing.dilate_python(kernel, (1, 1) + dilation)
        if fref is None:
            ref_res = topi.testing.conv2d_nchw_python(
                data.astype(out_dtype), dkernel.astype(out_dtype), 1, padding)
        else:
            ref_res = fref(data.astype(out_dtype), dkernel.astype(out_dtype))

        for target, ctx in ctx_list():
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(data, kernel)
            tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5, atol=1e-5)

    # depthwise conv2d
    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    run_test_conv2d("float32", "float32", 1, dshape, kshape,
                    padding=(1, 1), channels=32, groups=32, kernel_size=(3 ,3),
                    fref=lambda x, w: topi.testing.depthwise_conv2d_python_nchw(
                        x, w, (1, 1), "SAME"))

    # normal conv2d
    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    run_test_conv2d("float32", "float32", 1, dshape, kshape,
                    padding=(1, 1), channels=10, kernel_size=(3 ,3))
    # mixed precision
    run_test_conv2d("int8", "int32", 1, dshape, kshape,
                    padding=(1, 1), channels=10, kernel_size=(3 ,3))
    kshape = (10, 3, 1, 3)
    # mixed precision.
    run_test_conv2d("int8", "int32", 1, dshape, kshape,
                    padding=(0, 1), channels=10, kernel_size=(1 ,3))
    # dilated conv2d
    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    run_test_conv2d("float32", "float32", 1, dshape, kshape,
                    padding=(1, 1), channels=10, kernel_size=(3 ,3), dilation=(3, 3))


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


def test_conv2d_transpose_run():
    dshape = (1, 3, 18, 18)
    kshape = (3, 10, 3, 3)
    oshape = (1, 10, 37, 37)
    x = relay.var("x", shape=dshape)
    w = relay.var("w")
    y = relay.nn.conv2d_transpose(x, w,
                                  channels=10, kernel_size=(3,3), strides=(2,2),
                                  padding=(1,1), output_padding=(2, 2))
    func = relay.Function([x, w], y)
    dtype = "float32"
    data = np.random.uniform(size=dshape).astype(dtype)
    kernel = np.random.uniform(size=kshape).astype(dtype)
    c_np = topi.testing.conv2d_transpose_nchw_python(
        data, kernel, 2, 1)
    d_np = np.zeros(shape=oshape)
    d_np[:,:,0:c_np.shape[2],0:c_np.shape[3]] = c_np
    ref_res = d_np

    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(data, kernel)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5, atol=1e-5)



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


def _test_pool2d(opfunc, reffunc):
    n, c, h, w = tvm.var("n"), 10, 224, 224
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = opfunc(x, pool_size=(1, 1))
    assert "pool_size=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 10, 224, 224), "float32")
    # test execution
    dtype = "float32"
    dshape = (1, 3, 28, 28)
    x = relay.var("x", shape=dshape)
    y = opfunc(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
    func = relay.Function([x], y)
    data = np.random.uniform(size=dshape).astype(dtype)
    ref_res = reffunc(data.reshape(1,3,14,2,14,2), axis=(3,5))
    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5, atol=1e-5)


def _test_global_pool2d(opfunc, reffunc):
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
    # test execution
    dtype = "float32"
    dshape = (1, 1024, 7, 7)
    x = relay.var("x", shape=dshape)
    y = opfunc(x)
    func = relay.Function([x], y)
    data = np.random.uniform(size=dshape).astype(dtype)
    ref_res = reffunc(data, axis=(2,3), keepdims=True)
    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5, atol=1e-5)


def test_pool2d():
    _test_pool2d(relay.nn.max_pool2d, np.max)
    _test_pool2d(relay.nn.avg_pool2d, np.mean)
    _test_global_pool2d(relay.nn.global_max_pool2d, np.max)
    _test_global_pool2d(relay.nn.global_avg_pool2d, np.mean)


def test_avg_pool2d_no_count_pad():
    kh, kw = (4, 4)
    sh, sw = (2, 2)
    ph, pw = (2, 2)
    n = 1
    (ic, ih, iw) = (3, 28, 28)
    (oc, oh, ow) = (3, 15, 15)
    dshape = (n, ic, ih, iw)
    x = relay.var("x", shape=dshape)
    y = relay.nn.avg_pool2d(x,
                            pool_size=(kh, kw),
                            strides=(sw, sw),
                            padding=(ph, pw),
                            count_include_pad=False)
    func = relay.Function([x], y)
    dtype = "float32"
    a_np = np.random.uniform(low=0.001, size=(n, ic, ih, iw)).astype(dtype)
    pad_np = np.zeros(shape=(n, ic, ih+2*ph, iw+2*pw)).astype(dtype)
    no_zero = (range(n), range(ic), (range(ph, ih+ph)), (range(pw, iw+pw)))
    pad_np[np.ix_(*no_zero)] = a_np
    b_np = np.zeros(shape=(n, oc, oh, ow)).astype(dtype)
    for i in range(oh):
        for j in range(ow):
            pad_count = np.sum(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw] > 0, axis=(2,3))
            b_np[:,:,i,j] = np.sum(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw],
                                   axis=(2,3)) / np.maximum(pad_count, 1)
    ref_res = np.maximum(b_np, 0.0)
    data = a_np

    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5, atol=1e-5)


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

    shape = (1, 5, 10, 10)
    o_shape = (1, 500)
    dtype = "float32"
    x = relay.var("x", relay.TensorType(shape, dtype))
    z = relay.nn.batch_flatten(x)
    yy = relay.ir_pass.infer_type(z)
    assert yy.checked_type == relay.TensorType(o_shape, dtype)
    func = relay.Function([x], z)
    x_data = np.random.uniform(low=-1, high=1, size=shape).astype(dtype)
    ref_res = x_data.flatten().reshape(o_shape)

    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
        op_res2 = intrp2.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)

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

def test_pad_run():
    def _test_run(dtype):
        dshape = (4, 10, 7, 7)
        x = relay.var("x", shape=dshape)
        y = relay.nn.pad(x, ((1, 1), (2, 2), (3, 3), (4, 4)))
        func = relay.Function([x], y)
        data = np.random.uniform(size=dshape).astype(dtype)
        ref_res = np.pad(data, ((1, 1), (2, 2), (3, 3), (4, 4)), 'constant')
        for target, ctx in ctx_list():
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(data)
            tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5, atol=1e-5)

    _test_run('float32')
    _test_run('int32')

def test_lrn():
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = relay.var("x", shape=(n, c , h, w))
    y = relay.nn.lrn(x, size=10, axis=2, bias=0.5, alpha=.00001, beta=0.75)
    "alpha=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c , h, w))

    shape = (1, 5, 10, 10)
    dtype = "float32"
    x = relay.var("x", relay.TensorType(shape, dtype))
    size=5
    axis=1
    bias=0.5
    alpha=.00001
    beta=0.75
    z = relay.nn.lrn(x, size=size, axis=axis, bias=bias, alpha=alpha, beta=beta)
    yy = relay.ir_pass.infer_type(z)
    assert yy.checked_type == relay.TensorType(shape, dtype)
    func = relay.Function([x], z)
    x_data = np.random.uniform(low=-1, high=1, size=shape).astype(dtype)
    ref_res = topi.testing.lrn_python(x_data, size, axis, bias, alpha, beta)

    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
        op_res2 = intrp2.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)

def test_l2_normalize():
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = relay.var("x", shape=(n, c , h, w))
    y = relay.nn.l2_normalize(x, eps=0.001, axis=[1])
    "axis=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c , h, w))

    shape = (1, 5, 10, 10)
    dtype = "float32"
    x = relay.var("x", relay.TensorType(shape, dtype))
    eps=0.001
    axis=1
    z = relay.nn.l2_normalize(x, eps=0.001, axis=[axis])
    yy = relay.ir_pass.infer_type(z)
    assert yy.checked_type == relay.TensorType(shape, dtype)
    func = relay.Function([x], z)
    x_data = np.random.uniform(low=-1, high=1, size=shape).astype(dtype)
    ref_res = topi.testing.l2_normalize_python(x_data, eps, axis)

    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
        op_res2 = intrp2.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)


def batch_flatten(data):
    shape = data.shape
    target_dim = 1
    for i in range(len(shape) - 1):
        target_dim = target_dim * shape[i + 1]
    return np.reshape(data, (shape[0], target_dim))


def test_batch_flatten():
    t1 = relay.TensorType((5, 10, 5))
    x = relay.Var("x", t1)
    func = relay.Function([x], relay.nn.batch_flatten(x))

    data = np.random.rand(5, 10, 5).astype(t1.dtype)
    ref_res = batch_flatten(data)
    for target, ctx in ctx_list():
        intrp = relay.create_executor("graph", ctx=ctx, target=target)
        op_res = intrp.evaluate(func)(data)
        np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=0.01)


def _test_upsampling(layout, method):
    n, c, h, w = tvm.var("n"), 16, 32, 32
    scale = 2
    dtype = "float32"
    def get_shape():
        if layout == "NCHW":
            return (c, h, w), (c, h*scale, w*scale)
        else:
            return (h, w, c), (h*scale, w*scale, c)
    ishape, oshape = get_shape()
    x = relay.var("x", relay.TensorType((n,) + ishape, dtype))
    y = relay.nn.upsampling(x, scale=scale, layout=layout, method=method)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n,) + oshape, dtype)
    dshape = (1,) + ishape
    x = relay.var("x", shape=dshape)
    y = relay.nn.upsampling(x, scale=scale, layout=layout, method=method)
    func = relay.Function([x], y)
    data = np.random.uniform(size=dshape).astype(dtype)
    if method == "NEAREST_NEIGHBOR":
        ref = topi.testing.upsampling_python(data, scale, layout)
    else:
        ref = topi.testing.bilinear_resize_python(data, (h*scale, w*scale), layout)
    for target, ctx in ctx_list():
        executor = relay.create_executor("graph", ctx=ctx, target=target)
        out = executor.evaluate(func)(data)
        tvm.testing.assert_allclose(out.asnumpy(), ref, rtol=1e-5, atol=1e-5)


def test_upsampling():
    _test_upsampling("NCHW", "NEAREST_NEIGHBOR")
    _test_upsampling("NCHW", "BILINEAR")
    _test_upsampling("NHWC", "NEAREST_NEIGHBOR")
    _test_upsampling("NHWC", "BILINEAR")


if __name__ == "__main__":
    test_pool2d()
    test_avg_pool2d_no_count_pad()
    test_lrn()
    test_l2_normalize()
    test_conv2d_infer_type()
    test_upsampling_infer_type()
    test_flatten_infer_type()
    test_pad_infer_type()
    test_pad_run()
    test_conv2d_transpose_infer_type()
    test_conv2d_transpose_run()
    test_conv2d_run()
    test_batch_flatten()
    test_upsampling()
