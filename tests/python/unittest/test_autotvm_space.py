"""Test space definition primitives"""

import tvm
from tvm.autotvm.task.space import ConfigSpace, FallbackConfigEntity

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

    # test fallback
    cfg = FallbackConfigEntity()
    cfg.define_split('tile_n', cfg.axis(128), num_outputs=3)
    cfg.fallback_split('tile_n', [-1, 8, 4])
    assert cfg['tile_n'].size == [4, 8, 4]

    cfg = FallbackConfigEntity()
    cfg.define_split('tile_n', cfg.axis(49), num_outputs=3)
    cfg.fallback_split('tile_n', [-1, 8, 4])
    assert cfg['tile_n'].size == [7, 7, 1]

    cfg = FallbackConfigEntity()
    cfg.define_split('tile_n', cfg.axis(49), num_outputs=3)
    try:
        cfg.fallback_split('tile_n', [-1, 1, 0])
        assert False
    except RuntimeError:
        pass


if __name__ == '__main__':
    test_split()
