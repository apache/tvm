import tvm
import numpy as np

ctx = tvm.context('cpu', 0)
fadd1 = tvm.module.load('lib/test_addone.signed.so')

n = 10
x = tvm.nd.array(np.random.uniform(size=n).astype('float32'), ctx)
y = tvm.nd.array(np.zeros(n, dtype='float32'), ctx)
fadd1(x, y)

np.testing.assert_allclose(y.asnumpy(), x.asnumpy() + 1)
print("It works!")
