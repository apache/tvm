def pytorch_test():
    """This is a simple test function for PyTorch bridge

       It is not included as nosetests, because of its dependency on PyTorch

       User can directly run this script to verify correctness.
    """
    import numpy as np
    import tvm
    from tvm.contrib.pytorch import to_pytorch
    import torch

    x = torch.rand(56,56)
    y = torch.rand(56,56)
    z = x.mm(y)

    n = tvm.convert(56)
    X = tvm.placeholder((n,n), name='X')
    Y = tvm.placeholder((n,n), name='Y')

    k = tvm.reduce_axis((0, n), name='k')
    Z = tvm.compute((n,n), lambda i,j : tvm.sum(X[i,k]*Y[k,j], axis=k))
    s = tvm.create_schedule(Z.op)
    f = tvm.build(s, [X, Y, Z], target_host='llvm', name='f')

    f_pytorch = to_pytorch(f)
    z2 = torch.empty(56,56)
    f_pytorch(x, y, z2)
    np.testing.assert_allclose(z.numpy(), z2.numpy())

if __name__ == '__main__':
    pytorch_test()
