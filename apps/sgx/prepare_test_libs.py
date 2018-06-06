"""Script to prepare test_addone_sys.o"""

from __future__ import print_function
from os import path as osp

import tvm

CWD = osp.dirname(osp.abspath(osp.expanduser(__file__)))

KEY_HEAD_FOOT = '-----{} RSA PRIVATE KEY-----'
KEY = """ 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
""".strip()

def prepare_test_libs(base_path):
    n = tvm.var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1, name='B')
    s = tvm.create_schedule(B.op)
    s[B].parallel(s[B].op.axis[0])
    print(tvm.lower(s, [A, B], simple_mode=True))

    # Compile library in system library mode
    fadd_syslib = tvm.build(s, [A, B], 'llvm --system-lib')
    syslib_path = osp.join(base_path, 'test_addone_sys.o')
    fadd_syslib.save(syslib_path)

def main():
    out_dir = osp.join(CWD, 'lib')

    prepare_test_libs(out_dir)

    with open(osp.join(out_dir, 'enclave.pem'), 'w') as f_key:
        print(KEY_HEAD_FOOT.format('BEGIN'), file=f_key)
        print(KEY, file=f_key)
        print(KEY_HEAD_FOOT.format('END'), file=f_key)


if __name__ == '__main__':
    main()
