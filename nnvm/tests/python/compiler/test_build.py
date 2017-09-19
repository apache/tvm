import numpy as np

import tvm
import nnvm.symbol as sym
import nnvm.compiler
import nnvm.runtime

def test_compile():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.exp(y + x)
    shape = (10, 128)
    dtype = tvm.float32
    shape_dict = {"x": shape, "y": shape}

    graph, lib = nnvm.compiler.build(z, "llvm", shape_dict)
    m = nnvm.runtime.create(graph, lib, tvm.cpu(0))
    # get member functions
    set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
    na = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    nb = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    # set inputs
    set_input("x", na)
    set_input("y", nb)
    # execute
    run()
    # get outputs
    out = tvm.nd.empty(shape, dtype)
    get_output(0, out)
    np.testing.assert_allclose(
        out.asnumpy(), np.exp(na.asnumpy() + nb.asnumpy()))


def test_run():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.exp(y + x)
    shape = (10, 10)
    dtype = tvm.float32
    nx = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    ny = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    res = nnvm.compiler._run_graph(z, {"x": nx, "y": ny})
    np.testing.assert_allclose(
        res[0].asnumpy(), np.exp(nx.asnumpy() + ny.asnumpy()))


def test_precompute_prune():
    x = sym.Variable("x") + 1
    y = sym.Variable("y")
    z = y + x
    shape = (10, 10)
    dtype = tvm.float32
    nx = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    ny = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    params = {"x": nx}
    graph, pdict = nnvm.compiler.precompute_prune(z, params)
    pdict["y"] = ny
    res = nnvm.compiler._run_graph(z, pdict)
    np.testing.assert_allclose(
        res[0].asnumpy(), nx.asnumpy() + 1 + ny.asnumpy())


if __name__ == "__main__":
    test_compile()
    test_run()
    test_precompute_prune()
