import numpy as np
import tvm
from tvm.contrib import graph_runtime
import topi
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list

def test_relu():
    x = sym.Variable("x")
    y = sym.leaky_relu(x, alpha=0.3) - 0.2
    y = sym.relu(y)
    dtype = "float32"
    dshape = (1, 3, 32, 32)
    oshape = dshape
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        data = np.random.uniform(size=dshape).astype(dtype)
        m.run(x=data)
        data = (data < 0) * data * 0.3 + (data>0) * data - 0.2
        data = (data > 0) * data
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        np.testing.assert_allclose(out.asnumpy(), data, atol=1e-5, rtol=1e-5)


def test_exp():
    x = sym.Variable("x")
    y = sym.exp(x)
    dtype = "float32"
    dshape = (1, 3, 32, 32)
    oshape = dshape
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        data = np.random.uniform(size=dshape).astype(dtype)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        y_np = np.exp(data)
        np.testing.assert_allclose(out.asnumpy(), y_np, atol=1e-5, rtol=1e-5)


def test_log():
    x = sym.Variable("x")
    y = sym.log(x)
    dtype = "float32"
    dshape = (1, 3, 32, 32)
    oshape = dshape
    for target, ctx in ctx_list():
        with nnvm.compiler.build_config(opt_level=1):
            graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        data = np.random.uniform(size=dshape).astype(dtype)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        y_np = np.log(data)
        np.testing.assert_allclose(out.asnumpy(), y_np, atol=1e-5, rtol=1e-5)


def test_tanh():
    x = sym.Variable("x")
    y = sym.tanh(x)
    dtype = "float32"
    dshape = (1, 3, 32, 32)
    oshape = dshape
    for target, ctx in ctx_list():
        with nnvm.compiler.build_config(opt_level=1):
            graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        data = np.random.uniform(size=dshape).astype(dtype)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        y_np = np.sinh(data) / np.cosh(data)
        np.testing.assert_allclose(out.asnumpy(), y_np, atol=1e-5, rtol=1e-5)


def test_sigmoid():
    x = sym.Variable("x")
    y = sym.sigmoid(x)
    dtype = "float32"
    dshape = (1, 3, 32, 32)
    oshape = dshape
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        data = np.random.uniform(size=dshape).astype(dtype)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        y_np = 1.0 / (1.0 + np.exp(-data))
        np.testing.assert_allclose(out.asnumpy(), y_np, atol=1e-5, rtol=1e-5)


def test_softmax():
    x = sym.Variable("x")
    y = sym.softmax(x)
    dtype = "float32"
    dshape = (10, 1000)
    oshape = dshape
    for target, ctx in ctx_list():
        with nnvm.compiler.build_config(opt_level=1):
            graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        data = np.random.uniform(size=dshape).astype(dtype)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        y_np = topi.testing.softmax_python(data)
        np.testing.assert_allclose(out.asnumpy(), y_np, atol=1e-5, rtol=1e-5)


def test_log_softmax():
    x = sym.Variable("x")
    y = sym.log_softmax(x)
    dtype = "float32"
    dshape = (10, 1000)
    oshape = dshape
    for target, ctx in ctx_list():
        with nnvm.compiler.build_config(opt_level=1):
            graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        data = np.random.uniform(size=dshape).astype(dtype)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        y_np = topi.testing.log_softmax_python(data)
        np.testing.assert_allclose(out.asnumpy(), y_np, atol=1e-5, rtol=1e-5)


def test_dense():
    x = sym.Variable("x")
    y = sym.dense(x, units=3, name="dense")
    y = sym.flatten(y)
    dtype = "float32"
    shape = {
        "x" : (10, 100),
        "dense_weight" : (3, 100),
        "dense_bias" : (3,),
    }
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape)
        m = graph_runtime.create(graph, lib, ctx)
        x_np = np.random.uniform(size=shape["x"]).astype(dtype)
        w_np = np.random.uniform(size=shape["dense_weight"]).astype(dtype)
        b_np = np.random.uniform(size=shape["dense_bias"]).astype(dtype)
        res = tvm.nd.empty((10, 3))
        m.run(x=x_np, dense_weight=w_np, dense_bias=b_np)
        m.get_output(0, res)
        res_np = np.dot(x_np, w_np.T) + b_np
        np.testing.assert_allclose(
            res.asnumpy(), res_np, atol=1e-5, rtol=1e-5)


def test_batchnorm():
    x = sym.Variable("x")
    beta = sym.Variable("beta")
    gamma = sym.Variable("gamma")
    moving_var = sym.Variable("moving_var")
    moving_mean = sym.Variable("moving_mean")
    shape = (10, 20)
    eps = 1e-5
    dtype = "float32"
    y = sym.batch_norm(
        x, gamma, beta, moving_mean, moving_var, epsilon=eps)

    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, "llvm", {"x": shape})
        m = graph_runtime.create(graph, lib, tvm.cpu(0))
        x_np = np.random.uniform(size=shape).astype(dtype)
        mean_np = np.random.uniform(size=shape[1]).astype(dtype)
        var_np = np.random.uniform(size=shape[1]).astype(dtype)
        gamma_np = np.random.uniform(size=shape[1]).astype(dtype)
        beta_np = np.random.uniform(size=shape[1]).astype(dtype)
        res = tvm.nd.empty(shape)
        m.run(x=x_np, moving_mean=mean_np, moving_var=var_np,
              gamma=gamma_np, beta=beta_np)
        m.get_output(0, res)
        res_np = (x_np - mean_np) / np.sqrt(var_np + eps) * gamma_np + beta_np
        np.testing.assert_allclose(
            res.asnumpy(), res_np, atol=1e-5, rtol=1e-5)


def verify_concatenate(ishape, axis):
    x = [sym.Variable("x%d" % i) for i in range(len(ishape))]
    y = sym.concatenate(*x, axis=axis) + 1
    dtype = "float32"
    for target, ctx in ctx_list():
        # set input
        data = []
        for i, shape in enumerate(ishape):
            data.append(np.random.uniform(size=shape).astype(dtype))
        pdict = {"x%d" % i :  v for i, v in enumerate(data)}
        shape = {"x%d" % i :  v.shape for i, v in enumerate(data)}
        graph, lib, _ = nnvm.compiler.build(y, target, shape)
        m = graph_runtime.create(graph, lib, ctx)
        m.run(**pdict)
        out_np = np.concatenate(data, axis=axis) + 1
        out = m.get_output(0, tvm.nd.empty(out_np.shape))
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)

def test_concatenate():
    verify_concatenate([(2, 3, 4), (1, 3, 4)], axis=0)
    verify_concatenate([(2, 4), (2, 7)], axis=1)


def verify_split(ishape, indices_or_sections, axis):
    x = sym.Variable("x")
    y = sym.split(x, indices_or_sections=indices_or_sections, axis=axis)
    dtype = "float32"
    x_np = np.random.uniform(size=ishape).astype(dtype)
    res = np.split(x_np, indices_or_sections, axis=axis)
    for target, ctx in ctx_list():
        # set input
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": ishape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(x=x_np)
        for i, arr  in enumerate(res):
            out = m.get_output(i, tvm.nd.empty(arr.shape))
            np.testing.assert_allclose(out.asnumpy(), arr, atol=1e-5, rtol=1e-5)

def test_split():
    verify_split((2, 3), 2, axis=0)
    verify_split((5, 3), [3], axis=0)
    verify_split((5, 9, 3), [3, 4], axis=1)


def verify_squeeze(dshape, axis):
    x = sym.Variable("x")
    if axis:
        y = sym.squeeze(x, axis=axis)
    else:
        y = sym.squeeze(x)
    y = y + 1
    dtype = "float32"
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        m.run(x=data)
        out_np = np.squeeze(data.asnumpy(), axis=axis) + 1
        out = m.get_output(0, tvm.nd.empty(out_np.shape))
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)

def test_squeeze():
    verify_squeeze((1, 3, 2, 5), None)
    verify_squeeze((1, 3, 1), axis=0)
    verify_squeeze((1, 3, 2, 5, 1), axis=-1)

if __name__ == "__main__":
    test_split()
    test_concatenate()
    test_log_softmax()
    test_batchnorm()
    test_dense()
    test_relu()
    test_exp()
    test_log()
    test_tanh()
    test_sigmoid()
    test_softmax()
    test_squeeze()
