#!/usr/bin/env python3

import os.path as osp
import sys

import tvm
from tvm.contrib import cc


def main(target, out_dir):
    n = tvm.var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
    s = tvm.create_schedule(C.op)

    if target == 'cuda':
        bx, tx = s[C].split(C.op.axis[0], factor=64)
        s[C].bind(bx, tvm.thread_axis('blockIdx.x'))
        s[C].bind(tx, tvm.thread_axis('threadIdx.x'))

    fadd = tvm.build(s, [A, B, C], target, target_host='llvm', name='myadd')

    fadd.save(osp.join(out_dir, 'test_add.o'))
    if target == 'cuda':
        fadd.imported_modules[0].save(os.path.join(out_dir, 'test_add.ptx'))
    cc.create_shared(
        osp.join(out_dir, 'test_add.so'), [osp.join(out_dir, 'test_add.o')])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

