import nnvm
import numpy as np
import tvm
import topi
from tvm.contrib import graph_runtime
from nnvm import symbol as sym
from nnvm.compiler import graph_util, graph_attr
from nnvm.testing import ctx_list

def test_ewise_injective():
    def final_expression(x, y, n):
        y1 = sym.sqrt(y)
        p = sym.elemwise_add(x, y1)
        q = sym.elemwise_add(n, y1)
        r = sym.elemwise_add(q, y1)
        x1 = sym.sqrt(x)
        s = sym.elemwise_add(q, x1)
        t = sym.elemwise_add(r, s)
        u = sym.tanh(t)
        v = sym.elemwise_add(p, u)
        w = sym.elemwise_add(r, v)
        i = sym.tanh(s)
        j = sym.elemwise_add(w, i)
        k = sym.elemwise_add(j, y1)
        l = sym.elemwise_add(p, k)
        return l


    x = sym.Variable("x")
    y = sym.Variable("y")
    n = sym.Variable("n")
    z = sym.elemwise_add(x, sym.sqrt(y))
    i = sym.elemwise_add(n, sym.sqrt(y))
    p = sym.elemwise_add(i, sym.sqrt(y))
    q = sym.elemwise_add(i, sym.sqrt(x))
    r = sym.elemwise_add(p, q)
    s = sym.tanh(r)
    t = sym.elemwise_add(z, s)
    a = sym.elemwise_add(n, sym.sqrt(y))
    b = sym.elemwise_add(a, sym.sqrt(y))
    c = sym.elemwise_add(i, sym.sqrt(x))
    d = sym.elemwise_add(b, t)
    e = sym.tanh(c)
    f = sym.elemwise_add(d, e)
    g = sym.elemwise_add(f, sym.sqrt(y))
    k = sym.elemwise_add(z, g)
    dshape = (4,)
    shape_dict = {"x": dshape}
    dtype = "float32"
    target = "llvm"

    g = nnvm.graph.create(k)
    g2 = nnvm.graph.create(final_expression(x, y, n))
    graph_attr.set_shape_inputs(g, shape_dict)
    g1 = g.apply("InferShape").apply("SimplifyInference").apply("CommonSubExpression")
    # assert graph equals as expected
    graph_util.check_graph_equal(g1, g2)

    for target, ctx in ctx_list():
        x_np = np.array([1, 4, 9, 16]).astype("float32")
        y_np = np.array([4, 4, 4, 4]).astype("float32")
        n_np = np.array([9, 9, 9, 9]).astype("float32")
        params = {"y": y_np, "n": n_np}
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(k, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        params["x"] = x_np
        m["load_params"](nnvm.compiler.save_param_dict(params))
        m.run()
        out = m.get_output(0, tvm.nd.empty(dshape))
        np.testing.assert_allclose(
            out.asnumpy(),  np.array([23, 29, 39, 53]).astype("float32"),
            atol=1e-5, rtol=1e-5)

def test_ops_with_diffparams():
    def final_expression(x, y):
        y1 = sym.sqrt(y)
        p = sym.elemwise_add(x, y1)
        q = sym.leaky_relu(p, alpha = 0.3)
        r = sym.leaky_relu(p, alpha = 0.2)
        x1 = sym.sqrt(x)
        s = sym.elemwise_add(r, x1)
        t = sym.elemwise_add(s, p)
        v = sym.elemwise_add(q, t)
        return v


    x = sym.Variable("x")
    y = sym.Variable("y")
    s = sym.elemwise_add(x , sym.sqrt(y))
    z = sym.leaky_relu(s, alpha = 0.2)
    t = sym.elemwise_add(z , sym.sqrt(x))
    u = sym.elemwise_add(t , s)
    v = sym.elemwise_add(x , sym.sqrt(y))
    w = sym.leaky_relu(v, alpha = 0.3)
    k = sym.elemwise_add(w , u)
    dshape = (4,)
    shape_dict = {"x": dshape}
    dtype = "float32"
    target = "llvm"

    g = nnvm.graph.create(k)
    g2 = nnvm.graph.create(final_expression(x, y))
    graph_attr.set_shape_inputs(g, shape_dict)
    g1 = g.apply("InferShape").apply("SimplifyInference").apply("CommonSubExpression")
    # assert graph equals as expected
    graph_util.check_graph_equal(g1, g2)

    for target, ctx in ctx_list():
        x_np = np.array([1, 4, 9, 16]).astype("float32")
        y_np = np.array([4, 4, 4, 4]).astype("float32")
        params = {"y": y_np}
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(k, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        params["x"] = x_np
        m["load_params"](nnvm.compiler.save_param_dict(params))
        m.run()
        out = m.get_output(0, tvm.nd.empty(dshape))
        np.testing.assert_allclose(
            out.asnumpy(),  np.array([10, 20, 36, 58]).astype("float32"),
            atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    test_ewise_injective()
    test_ops_with_diffparams()
