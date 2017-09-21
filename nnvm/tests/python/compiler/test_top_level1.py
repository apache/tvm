import numpy as np

import tvm
import topi
import nnvm.symbol as sym
import nnvm.compiler
import nnvm.runtime

USE_GPU=True

def default_target():
    if USE_GPU:
        return 'cuda'
    else:
        return 'llvm'

def default_ctx():
    if USE_GPU:
        return tvm.gpu(0)
    else:
        return tvm.cpu(0)

def test_relu():
    x = sym.Variable("x")
    y = sym.relu(x)
    dtype = "float32"
    dshape = (1, 3, 32, 32)
    oshape = dshape
    graph, lib = nnvm.compiler.build(y, default_target(), {"x": dshape})
    m = nnvm.runtime.create(graph, lib, default_ctx())
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
    graph, lib = nnvm.compiler.build(y, default_target(), {"x": dshape})
    m = nnvm.runtime.create(graph, lib, default_ctx())
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
    graph, lib = nnvm.compiler.build(y, default_target(), {"x": dshape})
    m = nnvm.runtime.create(graph, lib, default_ctx())
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
    graph, lib = nnvm.compiler.build(y, default_target(), {"x": dshape})
    m = nnvm.runtime.create(graph, lib, default_ctx())
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
    graph, lib = nnvm.compiler.build(y, default_target(), {"x": dshape})
    m = nnvm.runtime.create(graph, lib, default_ctx())
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
    graph, lib = nnvm.compiler.build(y, default_target(), {"x": dshape})
    m = nnvm.runtime.create(graph, lib, default_ctx())
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


if __name__ == "__main__":
    test_relu()
    test_exp()
    test_log()
    test_tanh()
    test_sigmoid()
    test_softmax()
