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
# pylint: disable=invalid-name,unused-variable,unused-argument
"""GEMM schedules for Mali Bifrost"""
from tvm import te

from .transforms import tile_and_bind, tile_and_bind3d, interleave_transpose, transpose_interleave
from .. import utils


def decl_gemm(cfg, A, B):
    """Declare a single GEMM computation for Mali Bifrost GPUs

    Parameters
    ----------
    cfg : Config
        Schedule configuration

    A : tvm.te.Tensor
        2D Tensor, shape [n, k]

    B : tvm.te.Tensor
        2D Tensor, shape [k, m]

    Returns
    -------
    C : tvm.te.Tensor
        2D Tensor, shape [n, m]
    """

    cfg.define_knob("work_group_x", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64])
    cfg.define_knob("work_group_y", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64])
    cfg.define_knob("unroll_k_factor", [1, 2, 4])
    cfg.define_knob("A_interleave", [1, 4, 8, 16, 24, 32, 48, 64])
    cfg.define_knob("B_interleave", [1, 4, 8, 16, 32])
    cfg.define_knob("split_k_factor", [1, 4, 16])

    # Mutual k axis must be of equal extent
    assert utils.get_const_int(A.shape[1]) == utils.get_const_int(B.shape[0])
    n = A.shape[0]
    m = B.shape[1]
    k_size = utils.get_const_int(A.shape[1])
    unroll_gemm = cfg["split_k_factor"].val
    if unroll_gemm == 1:
        # No unrolling case must have the same set of tensors to keep scheduling consistent
        # Create identity tensors to take the place of A_unrolled, B_unrolled and R
        A_unrolled = te.compute((n, k_size), lambda i, j: A[i, j], name="A_unrolled")
        B_unrolled = te.compute((k_size, m), lambda i, j: B[i, j], name="B_unrolled")

        # Declare standard GEMM
        k = te.reduce_axis((0, A.shape[1]), name="k")
        C = te.compute(
            (n, m), lambda i, j: te.sum(A_unrolled[i, k] * B_unrolled[k, j], axis=k), name="C"
        )

        R = te.compute((n, m), lambda i, j: C[i, j], name="R")

    else:
        unrolled_k_size = k_size // unroll_gemm

        # Unroll the two input matrices along the shared k axis
        A_unrolled = te.compute(
            (unroll_gemm, n, unrolled_k_size),
            lambda b, i, j: A[i][unrolled_k_size * b + j],
            name="A_unrolled",
        )

        B_unrolled = te.compute(
            (unroll_gemm, unrolled_k_size, m),
            lambda b, i, j: B[unrolled_k_size * b + i][j],
            name="B_unrolled",
        )

        # Declare a batched GEMM
        k = te.reduce_axis((0, unrolled_k_size), name="k")
        C = te.compute(
            (unroll_gemm, n, m),
            lambda b, i, j: te.sum(A_unrolled[b][i][k] * B_unrolled[b][k][j], axis=k),
            name="C",
        )

        # Then declare a reduction to reduce the sub matrices
        k = te.reduce_axis((0, unroll_gemm), name="k")
        R = te.compute((n, m), lambda i, j: te.sum(C[k][i][j], axis=k), name="R")

    return R


def decl_batched_gemm(cfg, A, B):
    """Declare a batched GEMM computation for Mali Bifrost GPUs
    Parameters
    ----------
    cfg : Config
        Schedule configuration

    A : tvm.te.Tensor
        3D Tensor, shape [b, n, k]

    B : tvm.te.Tensor
        3D Tensor, shape [b, k, m]

    Returns
    -------
    C : tvm.te.Tensor
        3D Tensor, shape [b, n, m]

    """
    # Mutual b and k axis must be of equal extent
    assert utils.get_const_int(A.shape[2]) == utils.get_const_int(B.shape[1])
    assert utils.get_const_int(A.shape[0]) == utils.get_const_int(B.shape[0])

    cfg.define_knob("work_group_x", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64])
    cfg.define_knob("work_group_y", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64])
    cfg.define_knob("unroll_k_factor", [1, 2, 4])
    cfg.define_knob("A_interleave", [1, 4, 8, 16, 32, 64])
    cfg.define_knob("B_interleave", [1, 4, 8, 16, 32])

    n = A.shape[1]
    m = B.shape[2]
    k_size = utils.get_const_int(A.shape[2])
    b_size = utils.get_const_int(A.shape[0])

    # Declare a batched GEMM
    k = te.reduce_axis((0, k_size), name="k")
    C = te.compute(
        (b_size, n, m), lambda b, i, j: te.sum(A[b][i][k] * B[b][k][j], axis=k), name="C"
    )

    return C


def decl_winograd_gemm(cfg, A, B):
    """Declare a winograd GEMM for Mali Bifrost GPUs
    Winograd uses batched GEMM, however the input tensors are 4D
    This wraps decl_batched_gemm to provide it with 3D tensors

    Parameters
    ----------
    cfg : Config
        Schedule configuration

    A : tvm.te.Tensor
        4D Tensor, shape [a, a, n, k]

    B : tvm.te.Tensor
        4D Tensor, shape [a * a, k, m]

    Returns
    -------

    """
    alpha = utils.get_const_int(A.shape[0])
    n = utils.get_const_int(A.shape[2])
    k = utils.get_const_int(A.shape[3])

    A_3D = te.compute(
        (alpha * alpha, n, k), lambda b, i, j: A[b // alpha][b % alpha][i][j], name="A_3D"
    )

    C = decl_batched_gemm(cfg, A_3D, B)
    return A_3D, C


def schedule_gemm(cfg, s, A, B, C, batched=False, schedule_transforms=True):
    """Schedule GEMM, single and batched

    Parameters
    ----------
    cfg : Config
        Schedule configuration

    s : tvm.te.schedule.Schedule
        Operator schedule

    A : tvm.te.Tensor
        2D/3D Tensor, shape [n, k]/[b, n, k]

    B : tvm.te.Tensor
        2D/3D Tensor, shape [k, m]/[b, k, m]

    C : tvm.te.Tensor
        2D/3D Tensor, shape [n, m]/[b, n, m]

    batched : bool
        Whether the GEMM is batched

    Returns
    -------

    """
    block_size_x = 4
    block_size_y = 4
    warp_size_x = 2
    warp_size_y = 2

    work_group_x = cfg["work_group_x"].val
    work_group_y = cfg["work_group_y"].val
    k_unroll = cfg["unroll_k_factor"].val

    if not batched:
        y_index, x_index = (0, 1)
    else:
        y_index, x_index = (1, 2)

    trans_inter, A_transposed_interleaved = transpose_interleave(
        s, A, cfg["A_interleave"].val, y_index, x_index, [C], batched=batched
    )
    inter_trans, B_interleaved_transposed = interleave_transpose(
        s, B, cfg["B_interleave"].val, y_index, x_index, [C], batched=batched
    )

    if schedule_transforms:
        # Schedule A
        y, x = s[trans_inter].op.axis
        y, x, yi, xi = s[trans_inter].tile(y, x, 1, 8)
        s[trans_inter].unroll(yi)
        s[trans_inter].unroll(xi)
        tile_and_bind(s, trans_inter, y, x, 1, 4)

        # Schedule B
        y, x = s[inter_trans].op.axis
        xo, xi = s[inter_trans].split(x, 4)
        s[inter_trans].vectorize(xi)
        tile_and_bind(s, inter_trans, y, xo, 4, 4)

    # Schedule C
    CR_A = s.cache_read(A_transposed_interleaved, "local", [C])
    CR_B = s.cache_read(B_interleaved_transposed, "local", [C])
    CW_C = s.cache_write(C, "local")

    if not batched:
        y, x = s[C].op.axis
    else:
        z, y, x = s[C].op.axis
    y, x, yt, xt = s[C].tile(y, x, block_size_y, block_size_x)
    s[C].unroll(yt)
    s[C].vectorize(xt)
    # Tile the global work space to generate 'square' warps -> 2x2 for warp size of 4
    y, x, wy, wx = s[C].tile(y, x, warp_size_y, warp_size_x)
    x = s[C].fuse(x, wy, wx)
    if not batched:
        yo, xo, yi, xi = tile_and_bind(s, C, y, x, work_group_y, work_group_x)
    else:
        # For batched GEMM bind batch to z axis
        zo, yo, xo, zi, yi, xi = tile_and_bind3d(s, C, z, y, x, 1, work_group_y, work_group_x)

    s[CW_C].compute_at(s[C], xi)
    if not batched:
        y, x = s[CW_C].op.axis
    else:
        _, y, x = s[CW_C].op.axis
    y, x, yt, xt = s[CW_C].tile(y, x, block_size_y, block_size_x)
    k = s[CW_C].op.reduce_axis[0]
    s[CW_C].reorder(k, yt, xt)
    ko, ki = s[CW_C].split(k, k_unroll)
    s[CW_C].unroll(ki)
    s[CW_C].unroll(yt)
    s[CW_C].unroll(xt)

    if not batched:
        i, j = s[CR_A].op.axis
    else:
        _, i, j = s[CR_A].op.axis
    s[CR_A].reorder(j, i)
    s[CR_A].compute_at(s[CW_C], ki)
    s[CR_A].unroll(j)
    s[CR_A].vectorize(i)

    if not batched:
        i, j = s[CR_B].op.axis
    else:
        _, i, j = s[CR_B].op.axis
    s[CR_B].compute_at(s[CW_C], ki)
    s[CR_B].unroll(i)
    s[CR_B].vectorize(j)

    return trans_inter, inter_trans


def schedule_unrollable_gemm(cfg, s, A, B, C, R):
    """Schedule a GEMM that can be unrolled by a constant factor
    along its inner dimension

    Parameters
    ----------
    cfg : Config
        Schedule configuration

    s : tvm.te.schedule.Schedule
        Operator schedule

    A : tvm.te.Tensor
        2D/3D Tensor, shape [n, k]/[b, n, k]

    B : tvm.te.Tensor
        2D/3D Tensor, shape [k, m]/[b, k, m]

    C : tvm.te.Tensor
        2D/3D Tensor, shape [n, m]/[b, n, m]

    R : tvm.te.Tensor
        2D Tensor, shape [n, m]

    """
    # If the GEMM is 2D, no unrolling has taken place
    # Use non-batched GEMM schedule and inline identity matrices A, B and R
    if len(C.op.axis) == 2:
        s[A].compute_inline()
        s[B].compute_inline()
        schedule_gemm(cfg, s, A, B, C)
        s[R].compute_inline()

    # GEMM is 3D, use batched GEMM schedule, inline A and B and schedule R
    else:
        s[A].compute_inline()
        s[B].compute_inline()
        schedule_gemm(cfg, s, A, B, C, batched=True)

        CR_C = s.cache_read(C, "local", [R])

        y, x = s[R].op.axis
        xo, xi = s[R].split(x, 4)
        k = s[R].op.reduce_axis[0]
        s[R].reorder(k, xi)
        ko, ki = s[R].split(k, 4)
        s[R].unroll(xi)
        s[R].unroll(ki)
        tile_and_bind(s, R, y, xo, 1, 2)

        s[CR_C].compute_at(s[R], ko)
        _, y, x = s[CR_C].op.axis
        s[CR_C].unroll(y)
        s[CR_C].vectorize(x)


def get_unrollable_gemm_ops(R):
    """Get all GEMM operators from the final reduction
    This is a helper function to more easily get all the GEMM operations
    from an operator

    Parameters
    ----------
    R : tvm.te.Tensor
        Reduced tensor, final stage of GEMM

    Returns
    -------
    A_unrolled : tvm.te.Tensor
        Matrix A unrolled along k

    B_unrolled: tvm.te.Tensor
        Matrix B unrolled along k

    C : tvm.te.Tensor
        Result of batched GEMM

    R : tvm.te.Tensor
        Reduction of C, result of unrollable GEMM

    """
    C = R.op.input_tensors[0]
    A_unrolled, B_unrolled = C.op.input_tensors
    return A_unrolled, B_unrolled, C, R
