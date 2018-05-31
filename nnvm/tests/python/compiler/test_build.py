import numpy as np

import tvm
from tvm.contrib import graph_runtime
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.compiler.build_module import _run_graph, precompute_prune

def test_compile():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.exp(y + x)
    shape = (10, 128)
    dtype = tvm.float32
    shape_dict = {"x": shape, "y": shape}
    def verify(graph, lib):
        m = graph_runtime.create(graph, lib, tvm.cpu(0))
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
    m = graph_runtime.create(graph, lib, tvm.cpu(0))
    params["y"] = ny
    res = tvm.nd.empty(shape)
    m["load_params"](nnvm.compiler.save_param_dict(params))
    m.run()
    out = m.get_output(0, out=res)
    np.testing.assert_allclose(
        res.asnumpy(), nx.asnumpy() + 1 + ny.asnumpy() + na.asnumpy())


def test_dtypes():
    x = sym.Variable("x")
    y = sym.relu(x)
    dshape = (1, 3, 32, 32)
    oshape = dshape
    for dtype in ['float32', 'float64', 'int32', 'int16', 'int8', 'int64']:
        graph, lib, _ = nnvm.compiler.build(y, 'llvm', {"x": dshape}, dtype=dtype)
        m = graph_runtime.create(graph, lib, tvm.cpu())
        if 'float' in dtype:
          data = np.random.uniform(size=dshape).astype(dtype)
        elif 'int' in dtype:
          data = np.random.randint(-127, 127, dshape).astype(dtype)
        m.run(x=data)
        data = (data > 0) * data
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        np.testing.assert_allclose(out.asnumpy(), data, atol=1e-5, rtol=1e-5)

def test_compile_extra_lib():
    data = sym.Variable("data")
    net = sym.relu(data)
    net = sym.sqrt(net)
    out = sym.flatten(net)
    
    target = "cuda"
    extra_lib_target = "llvm"
    dshape = (1, 3, 56, 56)
    dtype = "float32"
    in_data = np.random.uniform(size=dshape).astype(dtype)
    opt_level = 2
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, _ = nnvm.compiler.build(out, target, {"data": dshape})
    m = graph_runtime.create(graph, lib, tvm.gpu(0))
    m.set_input("data", in_data)
    m.run()
    _, oshape = nnvm.compiler.graph_util.infer_shape(graph, shape={"data": dshape})
    expected_out = m.get_output(0, tvm.nd.empty(oshape, dtype))

    with nnvm.compiler.build_config(opt_level=opt_level, extra_lib_op="flatten", extra_lib_target=extra_lib_target):
        graph, lib, _, extra_libmod = nnvm.compiler.build(out, target, {"data": dshape})
    major_m = graph_runtime.create(graph, lib, tvm.gpu(0))
    major_m.set_input("data", in_data)
    major_m.run()
    major_out = major_m.get_output(0, tvm.nd.empty(dshape, dtype))
    extra_graph, extra_lib, _ = extra_libmod
    extra_m = graph_runtime.create(extra_graph, extra_lib, tvm.cpu())
    extra_input_name = extra_graph.symbol.list_input_names()[0]
    extra_m.set_input(extra_input_name, major_out)
    extra_m.run()
    final_out = extra_m.get_output(0, tvm.nd.empty(oshape, dtype))
    np.testing.assert_allclose(major_out.asnumpy(), final_out.asnumpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_precompute_prune()
    test_compile()
    test_run()
    test_dtypes()
    test_compile_extra_lib()
