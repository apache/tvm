import os.path as osp
import numpy as np
import tvm

CWD = osp.abspath(osp.dirname(__file__))


def main():
    ctx = tvm.context('cpu', 0)
    model = tvm.module.load(osp.join(CWD, 'build', 'enclave.signed.so'))
    inp = tvm.nd.array(np.ones((1, 3, 224, 224), dtype='float32'), ctx)
    out = tvm.nd.array(np.empty((1, 1000), dtype='float32'), ctx)
    model(inp, out)
    if abs(out.asnumpy().sum() - 1) < 0.001:
        print('It works!')
    else:
        print('It doesn\'t work!')
        exit(1)


if __name__ == '__main__':
    main()
