"""Utility functions for implementing Winograd convolutions"""
import numpy as np
from ..util import const_matrix


def winograd_transform_matrices(tile_size, out_dtype):
    """Compute the A, B, and G transform matrices for
    the tile size `m` as a `tvm.Expr`.
    """

    if tile_size not in (2, 4, 6):
        raise ValueError("Unsupported tile size for Winograd: {}".format(
            tile_size))
    if tile_size == 4:
        g_data = np.array(
            [
                [1 / 4.0, 0, 0],
                [-1 / 6.0, -1 / 6.0, -1 / 6.0],
                [-1 / 6.0, 1 / 6.0, -1 / 6.0],
                [1 / 24.0, 1 / 12.0, 1 / 6.0],
                [1 / 24.0, -1 / 12.0, 1 / 6.0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        b_data = np.array(
            [
                [4, 0, 0, 0, 0, 0],
                [0, -4, 4, -2, 2, 4],
                [-5, -4, -4, -1, -1, 0],
                [0, 1, -1, 2, -2, -5],
                [1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=out_dtype,
        )

        a_data = np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 1, 1],
                [1, -1, 1, -1],
                [1, 2, 4, 8],
                [1, -2, 4, -8],
                [0, 0, 0, 1],
            ],
            dtype=out_dtype,
        )

    elif tile_size == 6:
        g_data = np.array(
            [
                [1, 0, 0],
                [-2 / 9, -2 / 9, -2 / 9],
                [-2 / 9, 2 / 9, -2 / 9],
                [1 / 90, 1 / 45, 2 / 45],
                [1 / 90, -1 / 45, 2 / 45],
                [1 / 45, 1 / 90, 1 / 180],
                [1 / 45, -1 / 90, 1 / 180],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        assert np.dtype(out_dtype) == np.float32, "Only support floats in F(6x6, 3x3)"
        b_data = np.array(
            [
                [1, 0, -21 / 4, 0, 21 / 4, 0, -1, 0],
                [0, 1, 1, -17 / 4, -17 / 4, 1, 1, 0],
                [0, -1, 1, 17 / 4, -17 / 4, -1, 1, 0],
                [0, 1 / 2, 1 / 4, -5 / 2, -5 / 4, 2, 1, 0],
                [0, -1 / 2, 1 / 4, 5 / 2, -5 / 4, -2, 1, 0],
                [0, 2, 4, -5 / 2, -5, 1 / 2, 1, 0],
                [0, -2, 4, 5 / 2, -5, -1 / 2, 1, 0],
                [0, -1, 0, 21 / 4, 0, -21 / 4, 0, 1],
            ],
            dtype=out_dtype,
        ).T

        a_data = np.array(
            [
                [1, 1, 1, 1, 1, 32, 32, 0],
                [0, 1, -1, 2, -2, 16, -16, 0],
                [0, 1, 1, 4, 4, 8, 8, 0],
                [0, 1, -1, 8, -8, 4, -4, 0],
                [0, 1, 1, 16, 16, 2, 2, 0],
                [0, 1, -1, 32, -32, 1, -1, 1],
            ],
            dtype=out_dtype,
        ).T
    elif tile_size == 2:
        g_data = np.array(
            [
                [1, 0, 0],
                [1.0 / 2, 1.0 / 2, 1.0 / 2],
                [1.0 / 2, -1.0 / 2, 1.0 / 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        b_data = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, -1, 1],
                [-1, 1, 1, 0],
                [0, 0, 0, -1],
            ],
            dtype=out_dtype,
        )

        a_data = np.array(
            [[1, 0], [1, 1], [1, -1], [0, -1]],
            dtype=out_dtype
        )

    return (
        const_matrix(a_data, "A"),
        const_matrix(b_data, "B"),
        const_matrix(g_data, "G"),
    )
