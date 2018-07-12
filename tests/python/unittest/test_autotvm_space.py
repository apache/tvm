"""Test space definition primitives"""

import tvm
from tvm.autotvm.task.space import ConfigSpace

def gemm_func(cfg, N):
    A = tvm.placeholder((N, N), name='A')
    B = tvm.placeholder((N, N), name='B')

    k = tvm.reduce_axis((0, N), name='k')
    C = tvm.compute((N, N), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=[k]), name='C')

    s = tvm.create_schedule([C.op])

    y, x = s[C].op.axis

    cfg.define_split('tile_y', cfg.axis(y), num_outputs=2)
    cfg.define_split('tile_x', cfg.axis(x), num_outputs=2)

    return s, [A, B, C]

def test_split():
    cfg = ConfigSpace()

    gemm_func(cfg, 128)
    assert len(cfg) == 64
    assert len(cfg.space_map['tile_y']) == 8

if __name__ == '__main__':
    test_split()
