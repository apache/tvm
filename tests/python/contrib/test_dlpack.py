import tvm
import numpy as np
from tvm.contrib.dlpack import to_pytorch_func

def test():
    a = np.random.randn(1337)
    tvm_a = tvm.nd.array(a)
    np.testing.assert_equal(tvm.nd.from_dlpack(tvm_a.to_dlpack()).asnumpy(), a)

    try:
        import torch
        import torch.utils.dlpack

        x = torch.rand(56, 56)
        tvm_x = tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(x))
        np.testing.assert_equal(x.numpy(), tvm_x.asnumpy())
        y = tvm.nd.from_dlpack(tvm_x.to_dlpack())
        np.testing.assert_equal(y.asnumpy(), tvm_x.asnumpy())
        np.testing.assert_equal(torch.utils.dlpack.from_dlpack(y.to_dlpack()).numpy(), tvm_x.asnumpy())

        n = tvm.convert(137)
        xx = torch.rand(137,137)
        yy = torch.rand(137,137)
        zz2 = torch.empty(137,137)
        zz = xx.mm(yy)
        XX = tvm.placeholder((n,n), name='X')
        YY = tvm.placeholder((n,n), name='Y')

        k = tvm.reduce_axis((0, n), name='k')
        ZZ = tvm.compute((n,n), lambda i,j : tvm.sum(XX[i,k]*YY[k,j], axis=k))
        s = tvm.create_schedule(ZZ.op)
        f = tvm.build(s, [XX, YY, ZZ], target_host='llvm', name='f')

        f_pytorch = to_pytorch_func(f)
        zz2 = torch.empty(137,137)
        f_pytorch(xx, yy, zz2)
        np.testing.assert_allclose(zz.numpy(), zz2.numpy(), rtol=1e-6)

    except ImportError:
        pass


if __name__ ==  '__main__':
    test()
