"""Test flop calculation"""

import tvm
import numpy as np

from tvm.autotvm.task.task import compute_flop

def test_conv():
    for i in range(5):
        N, H, W, CO, CI, KH, KW = [np.random.randint(10, 32) for _ in range(7)]
        D = tvm.placeholder((N, CI, H, W))
        K = tvm.placeholder((CO, CI, KH, KW))

        KH = min(H, KH)
        KW = min(W, KW)

        ci = tvm.reduce_axis((0, CI))
        kh = tvm.reduce_axis((0, KH))
        kw = tvm.reduce_axis((0, KW))

        OH = (H - KH) + 1
        OW = (W - KW) + 1

        C = tvm.compute((N, CO, OH, OW), lambda n, co, h, w:
        tvm.sum(D[n][ci][h][w] * K[co][ci][h][w], axis=[ci, kh, kw]))

        s = tvm.create_schedule([C.op])

        assert compute_flop(s) == 2 * N * CO * OH * OW * CI * KH * KW

def test_pack_gemm():
    for i in range(5):
        N, L, M = [np.random.randint(10, 128) * 4 for _ in range(3)]
        A = tvm.placeholder((N, L))
        B = tvm.placeholder((M, L))
        k = tvm.reduce_axis((0, L))

        bn = 4
        A_pack = tvm.compute((N // bn, L, bn), lambda i, j, k: A[i * bn + k][j])
        B_pack = tvm.compute((M // bn, L, bn), lambda i, j, k: B[i * bn + k][j])
        C_pack = tvm.compute((N // bn, M // bn, bn, bn), lambda i, j, ii, jj:
        tvm.sum(A_pack[i, k, ii] * B_pack[j, k, jj], axis=[k]))
        C = tvm.compute((N, M), lambda i, j: C_pack[i // bn][j // bn][i % bn][j % bn])

        s = tvm.create_schedule([C.op])
        assert compute_flop(s) == 2 * N * L * M

def test_outer_dot():
    for i in range(5):
        N, M = [np.random.randint(10, 128) * 4 for _ in range(2)]
        A = tvm.placeholder((N,))
        B = tvm.placeholder((M,))

        C = tvm.compute((N, M), lambda i, j: A[i] * B[j])

        s = tvm.create_schedule([C.op])
        assert compute_flop(s) == N * M

def test_move():
    """No float number operation in simple move. So the estimator should raise an error """
    N = 1024

    A = tvm.placeholder((N,))
    C = tvm.compute((N,), lambda i: A[i])
    s = tvm.create_schedule([C.op])

    try:
        compute_flop(s)
        assert False
    except RuntimeError:
        pass

if __name__ == '__main__':
    test_conv()
    test_pack_gemm()
    test_outer_dot()
    test_move()
