import numpy as np

import tvm
import topi
import nnvm.symbol as sym
import nnvm.compiler
import nnvm.runtime

def ctx_list():
    res = [("llvm", tvm.cpu(0)), ("cuda", tvm.gpu(0))]
    return [x for x in res if x[1].exist]

def test_conv2d():
    x = sym.Variable("x")
    y = sym.conv2d(x, channels=10, kernel_size=(3, 3),
                   name="y", use_bias=False, padding=(1,1))
    dtype = "float32"
    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    oshape = (1, 10, 18, 18)
    shape_dict = {"x": dshape}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = nnvm.runtime.create(graph, lib, ctx)
        # get member functions
        set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        kernel = tvm.nd.array(np.random.uniform(size=kshape).astype(dtype))
        set_input("x", data)
        set_input("y_weight", kernel)
        # execute
        run()
        # get output
        out = tvm.nd.empty(oshape, dtype)
        get_output(0, out)
        c_np = topi.testing.conv2d_nchw_python(
            data.asnumpy(), kernel.asnumpy(), 1, 1)
        np.testing.assert_allclose(out.asnumpy(), c_np, rtol=1e-5)


def test_grouped_conv2d():
    x = sym.Variable("x")
    y = sym.conv2d(x, channels=32, kernel_size=(3, 3), groups=32,
                   name="y", use_bias=False, padding=(1,1))
    dtype = "float32"
    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    oshape = (1, 32, 18, 18)
    shape_dict = {"x": dshape}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = nnvm.runtime.create(graph, lib, ctx)
        # get member functions
        set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        kernel = tvm.nd.array(np.random.uniform(size=kshape).astype(dtype))
        set_input("x", data)
        set_input("y_weight", kernel)
        # execute
        run()
        # get output
        out = tvm.nd.empty(oshape, dtype)
        get_output(0, out)
        c_np = topi.testing.depthwise_conv2d_python_nchw(
            data.asnumpy(), kernel.asnumpy(), (1,1), 'SAME')
        np.testing.assert_allclose(out.asnumpy(), c_np, rtol=1e-5)


if __name__ == "__main__":
    test_conv2d()
    test_grouped_conv2d()
