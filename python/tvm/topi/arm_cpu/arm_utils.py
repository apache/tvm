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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Arm target utility functions"""

import tvm
from tvm.target import Target
from tvm.tir.expr import PrimExpr


def get_tiling_A(interleave_A, in_dtype, use_sme=False):
    """Compute the tiling information for matrix A in C=A*B,
    which corresponds to the im2col-transformed input matrix.

    The tiling information is chosen to maximize register usage during
    the tile computation.

    Please refer to:
    - https://discuss.tvm.apache.org/t/rfc-improve-quantized-convolution-performance-for-armv8-architectures # pylint: disable=line-too-long
    - https://discuss.tvm.apache.org/t/rfc-accelerate-quantized-convolution-through-dot-product
    - https://discuss.tvm.apache.org/t/rfc-improve-quantized-convolution-through-mmla-instruction
    - Conv2DGemmWeightTransformRel in src/relay/op/nn/convolution.h
     In order to have more information

    Parameters
    ----------
    interleave_A : bool
        determines if A is expected to be interleaved
    in_dtype : str
        input datatype
    use_sme : bool
        determines if SME operations on scalable vectors are expected

    Returns
    ----------
    tile_M: the output tile size of A on M axis (M = OH * OW)
    tile_K: the output tile size of A on K axis (K = KW * KH * IC)
    """
    target = Target.current(allow_none=False)
    if in_dtype in ["int8", "uint8"]:
        if target.features.has_matmul_i8:
            # If smmla/ummla is enabled, we are loading 8 rows from A. Each row
            # will contain 8 elements
            tile_M = 8
            tile_K = 8
        elif target.features.has_dotprod and interleave_A:
            # If dot product has been enabled, and we are interleaving A
            # tile size should be 8x4
            tile_M = 8
            tile_K = 4
        else:
            # If either there is no dot product or if we are using a native strategy
            # tile size should be 4x16
            tile_M = 4
            tile_K = 16
    elif use_sme:
        tile_M = 2 * tvm.tir.get_vscale_expr(in_dtype)
        if in_dtype == "float16":
            tile_K = tvm.tir.get_vscale_expr(in_dtype)
        else:
            tile_K = 2 * tvm.tir.get_vscale_expr(in_dtype)
    else:
        # In non-SME, non-quantized cases, A is not interleaved.
        # We are loading 4 rows from A.
        # Each row will contain 4 elements, along the dimension of reduction
        tile_M = 4
        tile_K = 4

    return tile_M, tile_K


def get_tiling_B_transformed(interleave_A, in_dtype, use_scalable_vectors=False, use_sme=False):
    """Compute the tiling information for matrix B', where B'
    is the tiled, interleaved (and transposed) version of matrix B in C=A*B.

    The tiling information is chosen to maximize register usage during the
    tile computation.

    Please refer to:
    - https://discuss.tvm.apache.org/t/rfc-improve-quantized-convolution-performance-for-armv8-architectures # pylint: disable=line-too-long
    - https://discuss.tvm.apache.org/t/rfc-accelerate-quantized-convolution-through-dot-product
    - https://discuss.tvm.apache.org/t/rfc-improve-quantized-convolution-through-mmla-instruction
    - Conv2DGemmWeightTransformRel in src/relay/op/nn/convolution.h
     In order to have more information

    Parameters
    ----------
    interleave_A : bool
        determines if A is expected to be interleaved
    in_dtype : str
        input datatype
    use_scalable_vectors : bool
        determines if operations on scalable vectors are expected
    use_sme : bool
        determines if SME operations on scalable vectors are expected


    Returns
    ----------
    tile_N: the output tile size of B' on N axis (N = OC)
    tile_K: the output tile size of B' on K axis (K = KW * KH * IC)
    """
    target = Target.current(allow_none=False)
    if in_dtype in ["int8", "uint8"]:
        if target.features.has_matmul_i8:
            # If smmla/ummla is available,  A must be interleaved.
            # Each load from B' will contain 8 elements
            # and we are loading 12 rows of B' (i.e., 12 columns of B)
            tile_N = 12
            tile_K = 8
        elif target.features.has_dotprod:
            # The number of tile rows of B' vary depending on the
            # strategy:
            # * If we are interleaving A, then we select 12 columns from B'(i.e.,
            #   12 rows from B).
            # * If we are not interleaving A, then we select 16 columns from B'(i.e.,
            #   16 rows from B).
            tile_N = 12 if interleave_A else 16

            # Dot product instruction groups 2 (u)int16x8 vectors in
            # groups of 4 and compute the dot product among those groups
            # This means that the number of columns in a tile of B' (i.e.,  the
            # rows of the original matrix B)  need to be 4.
            tile_K = 4
        else:
            # If no acceleration is available, A must be interleaved. In this case
            # we load 4 rows of B' (i.e., 4 columns of B). Each of them will contain 16 elements
            tile_N = 4
            tile_K = 16
    elif use_sme:
        tile_N = 2 * tvm.tir.get_vscale_expr(in_dtype)
        if in_dtype == "float16":
            tile_K = tvm.tir.get_vscale_expr(in_dtype)
        else:
            tile_K = 2 * tvm.tir.get_vscale_expr(in_dtype)
    # In non-SME, non-quantized cases, A is not interleaved.
    elif use_scalable_vectors:
        # Each load from B' contains 4 * scalable vectors (i.e. 4 * SVL columns from B)
        # We are loading 4 rows from B', in the dimension of reduction (i.e. 4 rows from B)
        tile_N = 4 * tvm.tir.get_vscale_expr(in_dtype)
        tile_K = 4
    elif in_dtype == "float16" and target.features.has_fp16_simd:
        # Each load from B' contains 32 elements (i.e. 32 columns from B)
        # We are loading 4 rows from B', in the dimension of reduction (i.e. 4 rows from B)
        tile_N = 32
        tile_K = 4
    else:
        # Each load from B' contains 16 elements (i.e. 16 columns from B)
        # We are loading 4 rows from B', in the dimension of reduction (i.e. 4 rows from B)
        tile_N = 16
        tile_K = 4

    return tile_N, tile_K


def get_conv2d_im2col_padding(M, K, tile_M, tile_K):
    """Compute the necessary padding for matrix A in C=A*B,
    which corresponds to the im2col-transformed input matrix.

    Parameters
    ----------
    M : int
        Number of rows in A = OH * OW
    K : int
        Number of columns in A = KW * KH * IC
    tile_M : int
             tile size of A on M axis
    tile_K : int
             tile size of A on K axis

    Returns
    ----------
    pad_M : padding for M axis
    pad_K : padding for K axis
    """
    pad_M = 0
    pad_K = 0

    if M % tile_M != 0:
        pad_M = tile_M - (M % tile_M)

    if K % tile_K != 0:
        pad_K = tile_K - (K % tile_K)

    return pad_M, pad_K


def pad_dim_to_multiple(dim: PrimExpr, multiple: PrimExpr):
    """
    Compute the padding required to reach specified multiple.

    Parameters
    ----------
    dim : PrimExpr
        Current size of the dim.
    multiple : PrimExpr
        Multiple to pad up to.

    Returns
    -------
    padded_dim : PrimExpr
        The new dim size.
    pad_value : PrimExpr
        The padding required.
    """
    pad_value = 0
    if dim % multiple != 0:
        pad_value = multiple - (dim % multiple)
    padded_dim = dim + pad_value
    return padded_dim, pad_value


def get_conv2d_weights_padding(N, K, tile_N, tile_K):
    """Compute the necessary padding for matrix B', where B'
    is the transformed version of matrix B in C=A*B.

    Parameters
    ----------
    N : int
        Number of columns in B = OC
    K : int
        Number of rows in B = KW * KH * IC
    tile_N : int
             tile size of B' on N axis
    tile_K : int
             tile size of B' on K axis

    Returns
    ----------
    pad_N : padding for N axis
    pad_K : padding for K axis
    """
    pad_N = 0
    pad_K = 0

    if N % tile_N != 0:
        pad_N = tile_N - (N % tile_N)

    # Tensorize will later make use of 4 tiles at once across the K axis so make sure we pad such
    # that K is multiple of 4
    K_multiplier = 4
    tile_K_multiplied = tile_K * K_multiplier
    K_misalignment = K % tile_K_multiplied

    if K_misalignment != 0:
        pad_K = tile_K_multiplied - K_misalignment

    return pad_N, pad_K
