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

""" Various tests related to the (WIP) support for having
one PrimFunc call another PrimFunc within the same IRModule.
"""

from typing import List
import pytest
import numpy as np

import tvm
import tvm.testing
import tvm.script
from tvm.script import tir as T

from tvm.contrib.hexagon.session import Session
from tvm.contrib.hexagon import allocate_hexagon_array
import test_hexagon.pytest_util as pytest_util
from test_hexagon.infrastructure import get_hexagon_target


# NOTE(cconvey): These pylint warnings should be re-enabled as TVM's pylint configuration matures.
# pylint: disable=missing-function-docstring,no-self-argument,invalid-name
# pylint: disable=redefined-outer-name,missing-class-docstring

# --------------------------------------------------------------------------------------------------
# Test parameters
# --------------------------------------------------------------------------------------------------

# The shape of the original (unsplit) tensors.
# We assume that each shape describes a non-empty 2D tensor.
original_shape = tvm.testing.parameter(
    # degenerate cases...
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    # arbitrary, provided for variety
    [5, 3],
    [3, 5],
)

# This dtype is arbitrary, but it must match the dtype that's hardcoded into the
# callee's function signature.  E.g., 'a_data: T.Ptr[T.int8]'.
#
# Hopefully over time we'll find a way to soften this limitation, at least for
# some approaches to PrimFunc-to-PrimFunc calls.
dtype = tvm.testing.parameter("int8")

# --------------------------------------------------------------------------------------------------
# Helper functions / definitions...
# --------------------------------------------------------------------------------------------------

HEXAGON_TARGET_ = get_hexagon_target("v69")

ENTRY_PRIMFUNC_NAME_ = "main"


def get_reference_input_tensor_(shape: list, dtype: str) -> np.array:
    a = pytest_util.create_populated_numpy_ndarray(
        shape, dtype, pytest_util.TensorContentFullRangeCOrder()
    )

    return a


def get_reference_output_tensor_(shape: list, dtype: str) -> np.array:
    return get_reference_input_tensor_(shape, dtype) + 1


def evaluate_ir_module_(
    hexagon_session: Session, shape: List, dtype: str, ir_mod: tvm.ir.module.IRModule
) -> np.array:
    reference_input_np = get_reference_input_tensor_(shape, dtype)
    reference_output_np = get_reference_output_tensor_(shape, dtype)

    hexagon_mod_local = tvm.build(
        ir_mod,
        target=get_hexagon_target("v69"),
        name=ENTRY_PRIMFUNC_NAME_,
    )

    hexagon_mod_remote = hexagon_session.load_module(hexagon_mod_local)

    input_data = allocate_hexagon_array(
        hexagon_session.device,
        data=reference_input_np,
    )

    output_data = allocate_hexagon_array(
        hexagon_session.device,
        tensor_shape=reference_output_np.shape,
        dtype=reference_output_np.dtype,
        data=np.full(shape, 0, dtype="int8"),
    )

    hexagon_mod_remote(input_data, output_data)

    output_data_np = output_data.numpy()
    tvm.testing.assert_allclose(reference_output_np, output_data_np)


# --------------------------------------------------------------------------------------------------
# Test cases...
# --------------------------------------------------------------------------------------------------


@tvm.testing.requires_hexagon
def test_baseline(
    hexagon_session: Session, original_shape: List, dtype: str
) -> tvm.ir.module.IRModule:
    dim0_size, dim1_size = original_shape

    @tvm.script.ir_module
    class AddOneBaseline:
        """
        Provides "add-one" functionality in a single, traditional PrimFunc.
        Used as a baseline for comparison / validation with other approaches.
        I.e., approaches that use various aspects of PrimFunc slicing and/or
        one PrimFunc calling into another.
        """

        @T.prim_func
        def main(a: T.handle, b: T.handle):
            # We exchange data between function by handles, which are similar to pointer.
            T.func_attr({"global_symbol": "main", "tir.noalias": True})

            A = T.match_buffer(a, original_shape, dtype=dtype)
            B = T.match_buffer(b, original_shape, dtype=dtype)

            for i in range(dim0_size):
                for j in range(dim1_size):
                    B[i, j] = A[i, j] + T.cast(1, dtype)

    evaluate_ir_module_(hexagon_session, original_shape, dtype, AddOneBaseline)


@tvm.testing.requires_hexagon
def test_pass_pointers(
    hexagon_session: Session, original_shape: List, dtype: str
) -> tvm.ir.module.IRModule:
    # Some notable requirements for this approach to intra-IRModule primfunc calls:
    #
    # - The specific dtype must he hardcoded into the callee's signature, e.g.
    #   'a_data: T.Ptr[T.int8]'.
    #
    # - The module's entry function must have a PrimFunc with the attribute
    #   "tir.is_entry_func": True.
    #   (This is related to having an IRModule with multiple PrimFuncs.)
    #
    # - The callee PrimFunc must have the "calling_conv": 3 attribute.
    #   (Where '3' is the number corresponding to 'tvm::CallingConv::kIntraModule'.)
    #   This ensures that the caller's 'A.data' argument, and the callee's 'a_data: T.ptr[T.int8]'
    #   parameter, both lower to 'uint8_t *', or something equivalent.
    #
    # - The callee must use 'T.buffer_decl' to describe the tile on which the callee
    #   shall operate.  As of this writing, there's no clear way to make this
    #   work with 'T.decl_buffer'.
    if dtype != "int8":
        pytest.skip(f"Unsupported dtype for this test: {dtype}")

    dim0_size, dim1_size = original_shape

    tile_shape = (dim1_size,)

    @tvm.script.ir_module
    class AddOnePassPointers:
        @T.prim_func
        def main(a: T.handle, b: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True, "tir.is_entry_func": True})

            A = T.match_buffer(a, original_shape, dtype=dtype)
            B = T.match_buffer(b, original_shape, dtype=dtype)

            for i in range(dim0_size):
                T.call_extern("", "callee", A.data, B.data, i)

        @T.prim_func
        def callee(a_data: T.Ptr[T.int8], b_data: T.Ptr[T.int8], i: T.int32):
            T.func_attr(
                {
                    "global_symbol": "callee",
                    "tir.noalias": True,
                    "calling_conv": 3,  # tvm::CallingConv::kIntraModule
                }
            )

            A_tile = T.buffer_decl(tile_shape, dtype, a_data, elem_offset=dim1_size * i)
            B_tile = T.buffer_decl(tile_shape, dtype, b_data, elem_offset=dim1_size * i)

            for j in range(dim1_size):
                B_tile[j] = A_tile[j] + T.cast(1, dtype)

    evaluate_ir_module_(hexagon_session, original_shape, dtype, AddOnePassPointers)


# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    tvm.testing.main()
