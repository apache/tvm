#!/usr/bin/env python3

"""Prepares a simple TVM library for testing."""

from os import path as osp
import sys

import tvm
from tvm.contrib import cc

def main():
    n = tvm.var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.create_schedule(C.op)
    s[C].parallel(s[C].op.axis[0])
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    obj_file = osp.join(sys.argv[1], 'test.o')
    tvm.build(s, [A, B, C], 'llvm').save(obj_file)
    cc.create_shared(osp.join(sys.argv[1], 'test.so'), [obj_file])

if __name__ == '__main__':
    main()
