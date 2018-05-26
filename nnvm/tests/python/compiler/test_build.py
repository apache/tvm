import numpy as np

import tvm
import nnvm.symbol as sym
import nnvm.compiler
import nnvm.runtime
from nnvm.compiler.build_module import _run_graph, precompute_prune

def test_compile():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.exp(y + x)
    shape = (10, 128)
    dtype = tvm.float32
    shape_dict = {"x": shape, "y": shape}
    def verify(graph, lib):
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

    graph, lib, _ = nnvm.compiler.build(z, "llvm", shape_dict)
    assert graph.index.num_nodes == 3
    verify(graph, lib)

    with nnvm.compiler.build_config(opt_level=0):
        graph, lib, _ = nnvm.compiler.build(z, "llvm", shape_dict)
        # print(graph.ir())
        assert graph.index.num_nodes == 4
        verify(graph, lib)

def test_run():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.exp(y + x)
    shape = (10, 10)
    dtype = tvm.float32
    nx = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    ny = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    res = _run_graph(z, {"x": nx, "y": ny})
    np.testing.assert_allclose(
        res[0].asnumpy(), np.exp(nx.asnumpy() + ny.asnumpy()))


def test_precompute_prune():
    x = sym.Variable("x") + 1
    a = sym.Variable("a")
    y = sym.Variable("y")
    z = y + x + a
    shape = (10, 10)
    dtype = tvm.float32
    nx = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    na = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    ny = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    params = {"x": nx, "a": na}
    graph, lib, params = nnvm.compiler.build(
        z, "llvm", shape={"y": ny.shape}, params=params)
    assert graph.index.num_nodes == 4
    m = nnvm.runtime.create(graph, lib, tvm.cpu(0))
    params["y"] = ny
    res = tvm.nd.empty(shape)
    m.run(**params)
    out = m.get_output(0, out=res)
    np.testing.assert_allclose(
        res.asnumpy(), nx.asnumpy() + 1 + ny.asnumpy() + na.asnumpy())


if __name__ == "__main__":
    test_precompute_prune()
    test_compile()
    test_run()
