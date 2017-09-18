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
    na = tvm.nd.array(np.ones(shape).astype(dtype))
    nb = tvm.nd.array(np.ones(shape).astype(dtype))
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

if __name__ == "__main__":
    test_compile()
