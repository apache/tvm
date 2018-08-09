import tvm
import numpy as np

def test():
    import torch
    import torch.utils.dlpack
    x = torch.rand(1)
    tvm_x = tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(x))
    np.testing.assert_equal(x.numpy(), tvm_x.asnumpy())

    #y = tvm.nd.from_dlpack(tvm_x.to_dlpack())
    print("finish from dlpack")
    print(y.asnumpy())
    print(tvm_x.shape)
    print(y.shape)
    np.testing.assert_equal(y.asnumpy(), tvm_x.asnumpy())

if __name__ ==  '__main__':
    test()
