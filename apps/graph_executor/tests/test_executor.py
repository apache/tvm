import tvm_graph as tg
import numpy as np
import tvm

def test_compile():
    x = tg.Variable('x')
    y = tg.Variable('y')
    z = tg.exp(y + x)
    shape = (10, 128)
    dtype = tvm.float32
    g = tg.build(z, "llvm",
                 shape={'x': shape,
                        'y': shape})
    m = tg.bind(g, tvm.cpu(0))
    # get member functions
    set_input, run, get_output = m['set_input'], m['run'], m['get_output']
    na = tvm.nd.array(np.ones(shape).astype(dtype))
    nb = tvm.nd.array(np.ones(shape).astype(dtype))
    # set inputs
    set_input('x', na)
    set_input('y', nb)
    # execute
    run()
    # get outputs
    out = tvm.nd.array(np.zeros(shape).astype(dtype))
    get_output(0, out)
    np.testing.assert_allclose(
        out.asnumpy(), np.exp(na.asnumpy() + nb.asnumpy()))

if __name__ == "__main__":
    test_compile()

