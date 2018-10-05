import os.path as osp
import numpy as np
import tvm

CWD = osp.abspath(osp.dirname(__file__))


def main():
    ctx = tvm.context('cpu', 0)
    model = tvm.module.load(osp.join(CWD, 'build', 'enclave.signed.so'))
    out = model()
    if out == 42:
        print('It works!')
    else:
        print('It doesn\'t work!')
        exit(1)


if __name__ == '__main__':
    main()
