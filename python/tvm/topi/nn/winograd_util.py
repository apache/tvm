# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
""" Utility functions for implementing Winograd convolutions
    [*] Fast Algorithms for Convolutional Neural Networks
        Andrew Lavin, Scott Gray
        https://arxiv.org/abs/1509.09308
        https://github.com/andravin/wincnn
"""

from operator import mul
from functools import reduce
import numpy as np
from tvm.contrib.pickle_memoize import memoize
from ..utils import const_matrix


# pylint: disable=invalid-name
def _cook_toom_convolution(a, n, r):
    """Compute Cook-Toom convolution A,B,G matrices"""

    def _F_m(a, n):
        f = lambda j, i: reduce(mul, ((a[i] - a[k] if k != i else 1) for k in range(0, n - 1)), 1)
        F = np.fromfunction(np.vectorize(f), (1, n - 1), dtype=int)
        F = np.diagflat(F)
        F = np.append(F, np.zeros((n - 1, 1), dtype=int), axis=1)
        f = lambda i, j: (1 if j == (n - 1) else 0)
        z = np.fromfunction(np.vectorize(f), (1, n), dtype=int)

        return np.append(F, z, axis=0)

    def _A_m(a, m, n):
        f = lambda i, j: a[i] ** j
        A = np.fromfunction(np.vectorize(f), (m - 1, n), dtype=int)
        f = lambda i, j: (1 if j == (n - 1) else 0)
        z = np.fromfunction(np.vectorize(f), (1, n), dtype=int)

        return np.append(A, z, axis=0)

    def _B_m(a, n):
        f = lambda j, i: reduce(mul, ((a[i] - a[k] if k != i else 1) for k in range(0, n - 1)), 1)
        Ff = np.fromfunction(np.vectorize(f), (1, n - 1), dtype=int)
        f = (
            lambda i, nth: (
                reduce(mul, [(np.poly1d([1, -a[k]]) if k != i else 1) for k in range(0, n - 1)], 1)
            ).coef[n - 1 - nth - 1]
            / Ff[0, i]
        )
        F = np.fromfunction(np.vectorize(f), (n - 1, n - 1), dtype=int)
        f = lambda i, j: -a[i] ** (n - 1)
        t = np.fromfunction(np.vectorize(f), (n - 1, 1), dtype=int)
        T = np.append(np.eye(n - 1), t, axis=1)

        return np.append(F.T.dot(T), np.array([np.eye(n)[n - 1]]), axis=0)

    alpha = n + r - 1

    f = _F_m(a, alpha)

    if f[0, 0] < 0:
        f[0, :] *= -1

    A = _A_m(a, alpha, n)

    G = _A_m(a, alpha, r).T
    G = G.dot(np.linalg.inv(f)).T

    B = _B_m(a, alpha)
    B = B.dot(f.T)

    return (A, B, G)


def _interpolation_points(degree):
    """Propose filter points"""

    assert 2 < degree < 18

    # Default interpolation lookup table
    #
    # [1] Error Analysis and Improving the Accuracy of Winograd Convolution for Deep Neural Networks
    #     Barbara Barabasz, Andrew Anderson, Kirk M. Soodhalter, David Gregg
    #     https://arxiv.org/abs/1803.10986
    #

    # pylint: disable=bad-whitespace,line-too-long
    in_pts = [
        #   {invalid}
        [],
        # 01 {E=4.63E-08 on conv2d  [1]}
        [],
        # 02 {E=7.65E-08 on F( 2,3) [1]}
        [0, -1, 1],
        # 03 {E=2.35E-07 on F( 3,3) [1]}
        [0, -1, 1, 1 / 2],
        # 04 {E=3.29E-07 on F( 4,3) [1]}
        [0, -1, 1, 1 / 2, -2],
        # 05 {E=6.81E-07 on F( 5,3) [1]}
        [0, -1, 1, 1 / 2, -2, -1 / 2],
        # 06 {E=8.79E-07 on F( 6,3) [1]}
        [0, -1, 1, 1 / 2, -1 / 2, 2, -2],
        # 07 {E=3.71E-06 on F( 7,3) [1]}
        [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4],
        # 08 {E=7.35E-06 on F( 8,3) [1]}
        [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 4],
        # 09 {E=2.20E-05 on F( 9,3) [1]}
        [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 3 / 4, -4 / 3],
        # 10 {E=3.22E-05 on F(10,3) [1]}
        [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 4, 3 / 4, -4 / 3],
        # 11 {E=1.09E-04 on F(11,3) [1]}
        [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 4, 3 / 4, -4 / 3, 1 / 4],
        # 12 {E=1.99E-04 on F(12,3) [1]}
        [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 4, 1 / 4, -3 / 4, 4 / 3, -4],
        # 13 {E=5.54E-04 on F(13,3) [1]}
        [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 4, 1 / 4, -3 / 4, 4 / 3, 3 / 4, -4 / 3],
        # 14 {E=8.80E-04 on F(14,3) [1]}
        [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 4, 1 / 4, -3 / 4, 4 / 3, -4, 3 / 4, -4 / 3],
        # 15 {E=1.07E-02 on F(15,3) [1]}
        [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 4, 1 / 4, -3 / 4, 4 / 3, -4, 2 / 3, -3 / 2, 3 / 2],
        # 16 {E=1.93E-02 on F(16,3) [1]}
        [
            0,
            -1,
            1,
            1 / 2,
            -1 / 2,
            2,
            -2,
            -1 / 4,
            4,
            1 / 4,
            -3 / 4,
            4 / 3,
            -4,
            2 / 3,
            -3 / 2,
            -2 / 3,
            3 / 2,
        ],
    ]  # pylint: enable=bad-whitespace,line-too-long

    return np.array(in_pts[degree - 1], dtype=np.float64)


@memoize("topi.nn.winograd_matrices", save_at_exit=False)
def winograd_transform_matrices(tile_size, kernel_size, out_dtype):
    """Compute the A, B, and G transform matrices for `tile_size` as a `tvm.Expr`."""
    if not 1 < tile_size < 9:
        raise ValueError("Unsupported tile size for Winograd: {}".format(tile_size))
    if not 2 < kernel_size < 8:
        raise ValueError("Unsupported kernel size for Winograd: {}".format(kernel_size))

    degree = tile_size + kernel_size - 2

    intp_pts = _interpolation_points(degree)
    A_data, B_data, G_data = _cook_toom_convolution(intp_pts, tile_size, kernel_size)

    return (
        const_matrix(A_data.astype(out_dtype), "A"),
        const_matrix(B_data.astype(out_dtype), "B"),
        const_matrix(G_data.astype(out_dtype), "G"),
    )
