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
    # get outputs
    out = tvm.nd.empty(oshape, dtype)
    get_output(0, out)
    y_np = topi.testing.softmax_python(data.asnumpy())
    np.testing.assert_allclose(out.asnumpy(), y_np, rtol=1e-5)


if __name__ == "__main__":
    test_softmax()
