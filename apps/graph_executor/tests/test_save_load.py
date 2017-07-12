import tvm_graph as tg
import numpy as np
import tvm

def test_save_load():
    shape = (10, 128)
    dtype = tvm.float32
    na = tvm.nd.array(np.ones(shape).astype(dtype))
    nb = tvm.nd.array(np.ones(shape).astype(dtype))

    x = tg.Variable('x')
    y = tg.Variable('y')
    z = tg.exp(y + x)

    g = tg.build(z, "llvm", shape={'x': shape, 'y': shape})
    m0 = tg.bind(g, tvm.cpu(0))
    set_input0, run0, get_output0 = m0['set_input'], m0['run'], m0['get_output']
    set_input0(0, na)
    set_input0(1, nb)
    run0()
    out0 = tvm.nd.array(np.zeros(shape).astype(dtype))
    get_output0(0, out0)

    tg.save_params('test.params', {'x': na, 'y': nb})

    # create another executor
    m1 = tg.bind(g, tvm.cpu(0))
    load_params1 = m1['load_params']
    load_params1('test.params')

    run1, get_output1 = m1['run'], m1['get_output']
    run1()
    out1 = tvm.nd.array(np.zeros(shape).astype(dtype))
    get_output1(0, out1)
    np.testing.assert_allclose(out0.asnumpy(), out1.asnumpy())

if __name__ == "__main__":
    test_save_load()
