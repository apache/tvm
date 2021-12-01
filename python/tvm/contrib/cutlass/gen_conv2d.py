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
# pylint: disable=invalid-name
"""Conv2d kernel generator and profiler for CUTLASS."""
from .conv2d_operation import Conv2dOperation, EmitConv2dInstance
from .gen_gemm import CutlassGemmProfiler
from .library import (
    EpilogueFunctor,
    SwizzlingFunctor,
    TensorDescription,
    LayoutType,
    ConvKind,
    StrideSupport,
    IteratorAlgorithm,
)


def create_conv2d_operator(
    tile_descriptions,
    data_type,
    alignment_constraints,
    swizzling_functor=SwizzlingFunctor.Identity4,
):
    """Exhaustively instantiate all kernels from a given configuration."""
    ret = []

    kernel_emitter = EmitConv2dInstance()

    element_a, element_b, element_c, element_epilogue = data_type
    iterator_algorithms = [IteratorAlgorithm.Optimized]

    layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    for tile in tile_descriptions:
        for alignment in alignment_constraints:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment)
            B = TensorDescription(element_b, layout[1], alignment)
            C = TensorDescription(element_c, layout[2], alignment_c)

            swizzling_functor_ = swizzling_functor

            for iterator_algorithm in iterator_algorithms:
                op_entry = {}

                op = Conv2dOperation(
                    ConvKind.Fprop,
                    iterator_algorithm,
                    tile.minimum_compute_capability,
                    tile,
                    A,
                    B,
                    C,
                    element_epilogue,
                    StrideSupport.Strided,
                    EpilogueFunctor.LinearCombination,
                    swizzling_functor_,
                )

                # TODO(masahi): Add profiler source here
                op_entry["opdef"] = kernel_emitter.emit(op)
                op_entry["op"] = op
                op_entry["name"] = op.procedural_name()
                op_entry["runtime"] = 9999999

                # fused ops
                for epilogue, opdef in zip(
                    [
                        EpilogueFunctor.LinearCombinationBias,
                        EpilogueFunctor.LinearCombinationRelu,
                    ],
                    ["opdef_bias", "opdef_bias_relu"],
                ):
                    op = Conv2dOperation(
                        ConvKind.Fprop,
                        iterator_algorithm,
                        tile.minimum_compute_capability,
                        tile,
                        A,
                        B,
                        C,
                        element_epilogue,
                        StrideSupport.Strided,
                        epilogue,
                        swizzling_functor_,
                    )

                    op_entry[opdef] = kernel_emitter.emit(op)

                ret.append(op_entry)

    return ret


class CutlassConv2DProfiler:
    """Profile all candidate kernels and select the best one."""

    def __init__(self, sm, cutlass_path, binary_path):
        self.gemm_profiler = CutlassGemmProfiler(sm, cutlass_path, binary_path)
        self.sm = sm

    def get_default(self, out_dtype):
        gemm_profile_result = self.gemm_profiler.get_default(out_dtype)
        tile_description = gemm_profile_result["tile_description"]
        alignment = gemm_profile_result["alignment"]
        data_type = gemm_profile_result["data_type"]
        return create_conv2d_operator([tile_description], data_type, [alignment])[0]

    def profile(
        self, d_shape, w_shape, out_shape, out_dtype, profile_all=True, use_multiprocessing=False
    ):
        """Profile and select the best kernel from candidate kernels.
        If profile_all is False, return immediately after the first applicable kernel is found.
        If use_multiprocessing is True, compile all profiler executables in parallel.
        """
        B, H, W, C = d_shape
        K, R, S, _ = w_shape
        _, P, Q, _ = out_shape

        M = B * H * W
        K = R * S * C
        N = B * P * Q

        gemm_profile_result = self.gemm_profiler.profile(
            M, K, N, out_dtype, profile_all=profile_all, use_multiprocessing=use_multiprocessing
        )

        tile_description = gemm_profile_result["tile_description"]
        alignment = gemm_profile_result["alignment"]
        data_type = gemm_profile_result["data_type"]

        return create_conv2d_operator([tile_description], data_type, [alignment])[0]
