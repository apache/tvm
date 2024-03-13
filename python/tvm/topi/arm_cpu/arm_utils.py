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

from tvm.target import Target


def get_tiling_B_transformed(interleave_A, in_dtype):
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
    # In non-quantized cases, A is not interleaved.
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
