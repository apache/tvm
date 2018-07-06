"""Script to prepare test_addone_sys.o"""

from os import path as osp

import tvm

CWD = osp.dirname(osp.abspath(osp.expanduser(__file__)))


def main():
    out_dir = osp.join(CWD, 'lib')

    n = tvm.var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1, name='B')
    s = tvm.create_schedule(B.op)
    s[B].parallel(s[B].op.axis[0])
    print(tvm.lower(s, [A, B], simple_mode=True))

    # Compile library in system library mode
    fadd_syslib = tvm.build(s, [A, B], 'llvm --system-lib')
    fadd_syslib.save(osp.join(out_dir, 'test_addone_sys.o'))


if __name__ == '__main__':
    main()
