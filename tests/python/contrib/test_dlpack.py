import tvm
import numpy as np

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

    except ImportError:
        pass


if __name__ ==  '__main__':
    test()
