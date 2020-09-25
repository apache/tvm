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


def is_dotprod_available():
    """ Checks whether the hardware has support for fast Int8 arithmetic operations. """
    target = tvm.target.Target.current(allow_none=False)
    return "+v8.2a" in target.mattr and "+dotprod" in target.mattr


def is_aarch64_arm():
    """ Checks whether we are compiling for an AArch64 target. """
    target = tvm.target.Target.current(allow_none=False)
    return "aarch64" in target.attrs.get("mtriple", "")


def get_tiling_B_interleaved_t(interleave_A):
    """Compute the tiling information for matrix B', where B'
    is the transposed and interleaved version of matrix B in C=A*B.

    The tiling information is chosen to maximize register usage during the
    tile computation.

    Please refer to:
        - https://discuss.tvm.apache.org/t/rfc-accelerate-quantized-convolution-through-dot-product
        - Conv2DGemmWeightTransformRel in src/relay/op/nn/convolution.h
     In order to have more information

    Parameters
    ----------
    interleave_A: bool
                  determines if A is expected to be interleaved

    Returns
    ----------
    tile_rows_B: the output tile rows of B'
    tile_cols_B: the output tile columns of B'
    """
    if is_dotprod_available():
        # The number of tile rows of B' vary depending on the
        # strategy:
        # * If we are interleaving A, then we select 12 columns from B'(i.e.,
        #   12 rows from B).
        # * If we are not interleaving A, then we select 16 columns from B'(i.e.,
        #   16 rows from B).
        tile_rows_B = 12 if interleave_A else 16

        # Dot product instruction groups 2 (u)int16x8 vectors in
        # groups of 4 and compute the dot product among those groups
        # This means that the number of columns in a tile of B' (i.e.,  the
        # rows of the original matrix B)  need to be 4.
        tile_cols_B = 4
    else:
        # If dot product is not available, A must be interleaved. In this case
        # we load 4 rows of B' (i.e., 4 columns of B). Each of them will contain 16 elements
        tile_rows_B = 4
        tile_cols_B = 16

    return tile_rows_B, tile_cols_B
