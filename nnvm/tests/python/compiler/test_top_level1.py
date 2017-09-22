import numpy as np

import tvm
import topi
import nnvm.symbol as sym
import nnvm.compiler
import nnvm.runtime

def ctx_list():
    res = [("llvm", tvm.cpu(0)), ("cuda", tvm.gpu(0))]
    return [x for x in res if x[1].exist]


def test_relu():
    x = sym.Variable("x")
    y = sym.relu(x)
    dtype = "float32"
    dshape = (1, 3, 32, 32)
    oshape = dshape
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = nnvm.runtime.create(graph, lib, ctx)
        # get member functions
        set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        set_input("x", data)
        # execute
        run()
        # get output
        out = tvm.nd.empty(oshape, dtype)
        get_output(0, out)
        y_np = np.maximum(data.asnumpy(), 0.0)
        np.testing.assert_allclose(out.asnumpy(), y_np, atol=1e-5, rtol=1e-5)


def test_exp():
    x = sym.Variable("x")
    y = sym.exp(x)
    dtype = "float32"
    dshape = (1, 3, 32, 32)
    oshape = dshape
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = nnvm.runtime.create(graph, lib, ctx)
        # get member functions
        set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        set_input("x", data)
        # execute
        run()
        # get output
        out = tvm.nd.empty(oshape, dtype)
        get_output(0, out)
        y_np = np.exp(data.asnumpy())
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
        m = nnvm.runtime.create(graph, lib, ctx)
        # get member functions
        set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        set_input("x", data)
        # execute
        run()
        # get output
        out = tvm.nd.empty(oshape, dtype)
        get_output(0, out)
        y_np = np.log(data.asnumpy())
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
        m = nnvm.runtime.create(graph, lib, ctx)
        # get member functions
        set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        set_input("x", data)
        # execute
        run()
        # get output
        out = tvm.nd.empty(oshape, dtype)
        get_output(0, out)
        y_np = np.sinh(data.asnumpy()) / np.cosh(data.asnumpy())
        np.testing.assert_allclose(out.asnumpy(), y_np, atol=1e-5, rtol=1e-5)


def test_sigmoid():
    x = sym.Variable("x")
    y = sym.sigmoid(x)
    dtype = "float32"
    dshape = (1, 3, 32, 32)
    oshape = dshape
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = nnvm.runtime.create(graph, lib, ctx)
        # get member functions
        set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        set_input("x", data)
        # execute
        run()
        # get output
        out = tvm.nd.empty(oshape, dtype)
        get_output(0, out)
        y_np = 1.0 / (1.0 + np.exp(-data.asnumpy()))
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
        m = nnvm.runtime.create(graph, lib, ctx)
        # get member functions
        set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        set_input("x", data)
        # execute
        run()
        # get output
        out = tvm.nd.empty(oshape, dtype)
        get_output(0, out)
        y_np = topi.testing.softmax_python(data.asnumpy())
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
    graph, lib, _ = nnvm.compiler.build(y, "llvm", shape)
    m = nnvm.runtime.create(graph, lib, tvm.cpu(0))
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
        m = nnvm.runtime.create(graph, lib, tvm.cpu(0))
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


if __name__ == "__main__":
    test_batchnorm()
    test_dense()
    test_relu()
    test_exp()
    test_log()
    test_tanh()
    test_sigmoid()
    test_softmax()
