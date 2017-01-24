import tvm
import numpy as np



def test_function():
    ctx = tvm.cpu(0)
    x = np.random.randint(0, 10, size=(3, 4))
    x = np.array(x)
    y = tvm.nd.array(x, ctx=ctx)

    f = tvm.codegen.DummyHelloFunction()
    f(y, 10)


if __name__ == "__main__":
    test_function()
