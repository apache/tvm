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
"""GEMM kernel generator and profiler for CUTLASS."""
import os
import pickle
from functools import partial

from .gemm_operation import EmitGemmInstance, GemmOperation
from .gemm_profiler import GemmProfilerEmitter
from .gen_tensor_op import EPILOGUE_MAP, GENERATOR_FUNC_TABLE, ProfilerEngine
from .library import (
    DataType,
    DataTypeTag,
    EpilogueFunctor,
    LayoutType,
    SwizzlingFunctor,
    TensorDescription,
)


def create_gemm_operator_with_epilogue(
    op_type,
    tile_description,
    data_type,
    alignment,
    swizzling_functor,
    batched=False,
    layout_b=LayoutType.ColumnMajor,
):
    """
    Instantiate a cutlass kernel from the given configuration,
    along with the epilouge functor
    """
    element_a, element_b, element_c, element_epilogue = data_type

    A = TensorDescription(element_a, LayoutType.RowMajor, alignment)
    B = TensorDescription(element_b, layout_b, alignment)
    C = TensorDescription(element_c, LayoutType.RowMajor, alignment)

    if batched:
        swizzling_functor = SwizzlingFunctor.Batched

    if "residual" in op_type:
        if "hardswish" in op_type:
            activation = "cutlass::epilogue::thread::HardSwish"
        elif "silu" in op_type:
            activation = "cutlass::epilogue::thread::SiLu"
        elif "sigmoid" in op_type:
            activation = "cutlass::epilogue::thread::Sigmoid"
        elif "gelu" in op_type:
            activation = "cutlass::epilogue::thread::GELU"
        elif "relu" in op_type:
            activation = "cutlass::epilogue::thread::ReLu"
        else:
            activation = "cutlass::epilogue::thread::Identity"

        binary_op = "cutlass::multiplies" if "residual_multiply" in op_type else "cutlass::plus"
        unary_op = (
            "cutlass::epilogue::thread::ReLu"
            if op_type.endswith("relu")
            else "cutlass::epilogue::thread::Identity"
        )
        residual_block_info = {
            "activation": activation,
            "binary_op": binary_op,
            "unary_op": unary_op,
        }
        epilogue = EpilogueFunctor.LinearCombinationResidualBlock
        no_beta_scaling = False
    else:
        residual_block_info = None
        epilogue, no_beta_scaling = EPILOGUE_MAP[op_type]

    op = GemmOperation(
        tile_description.minimum_compute_capability,
        tile_description,
        A,
        B,
        C,
        element_epilogue,
        epilogue,
        swizzling_functor,
    )

    return (
        op.procedural_name(),
        EmitGemmInstance().emit(
            op,
            no_beta_scaling=no_beta_scaling,
            batched=batched,
            residual_block_info=residual_block_info,
        ),
    )


def enumerate_gemm_operators(
    tile_descriptions,
    data_type,
    alignment_constraints,
    swizzling_functor=SwizzlingFunctor.Identity8,
    layout_b=LayoutType.ColumnMajor,
):
    """Exhaustively instantiate all kernels from a given configuration."""
    ret = []
    kernel_emitter = EmitGemmInstance()
    profiler_emitter = GemmProfilerEmitter()

    element_a, element_b, element_c, element_epilogue = data_type

    for tile_description in tile_descriptions:
        for alignment in alignment_constraints:
            A = TensorDescription(element_a, LayoutType.RowMajor, alignment)
            B = TensorDescription(element_b, layout_b, alignment)
            C = TensorDescription(element_c, LayoutType.RowMajor, alignment)

            if element_c == DataType.s32 and A.alignment == 1:
                tile_description.threadblock_shape[0] = min(
                    tile_description.threadblock_shape[0], 128
                )
                tile_description.threadblock_shape[1] = min(
                    tile_description.threadblock_shape[1], 128
                )

            op = GemmOperation(
                tile_description.minimum_compute_capability,
                tile_description,
                A,
                B,
                C,
                element_epilogue,
                EpilogueFunctor.LinearCombination,
                swizzling_functor,
            )

            src = profiler_emitter.emit(
                op.procedural_name(),
                kernel_emitter.emit(op, batched=False),
                DataTypeTag[element_a],
                DataTypeTag[element_b],
                DataTypeTag[element_c],
                op.leading_dim(),
            )

            ret.append(
                {
                    "src": src,
                    "op": op,
                    "name": op.procedural_name(),
                    "tile_description": tile_description,
                    "alignment": alignment,
                    "data_type": data_type,
                    "swizzle_functor": swizzling_functor,
                }
            )

    return ret


# TODO(masahi): A sensible way to pick reasonable default kernels
DEFAULT_KERNELS = {
    75: {
        ("float16", "float16"): "cutlass_tensorop_h1688gemm_128x64_32x2_tn_align1",
        ("float16", "float32"): "cutlass_tensorop_s1688gemm_f16_64x64_32x2_tn_align1",
    },
    # align1 variants do not seem to be available for sm80
    80: {
        ("float16", "float16"): "cutlass_tensorop_h1688gemm_128x64_32x2_tn_align1",
        ("float16", "float32"): "cutlass_tensorop_s1688gemm_f16_64x64_32x2_tn_align1",
        # two kernels for tf32 and 3xtf32
        ("float32", "float32"): (
            "cutlass_tensorop_s1688gemm_128x64_32x3_tn_align1",
            "cutlass_tensorop_s1688gemm_64x64_16x3_tn_align1",
        ),
    },
}


class CutlassGemmProfiler:
    """Profile all candidate kernels and select the best one."""

    def __init__(self, sm, cutlass_path, binary_path):
        assert sm in GENERATOR_FUNC_TABLE and sm in DEFAULT_KERNELS, f"sm{sm} not supported yet."
        self.engine = ProfilerEngine(sm, cutlass_path, binary_path)
        self.sm = sm
        self.cache_path = os.path.join(binary_path, "cutlass_gemm_cache.pickle")
        if os.path.exists(self.cache_path):
            self.cache = pickle.load(open(self.cache_path, "rb"))
        else:
            self.cache = {}

    def get_default(
        self,
        op_type,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        use_3xtf32=True,
        batched=False,
        layout_b=LayoutType.ColumnMajor,
    ):
        """Return the default kernel for the requested architecture.
        For now, the default kernel was picked arbitrary.
        """
        ops = GENERATOR_FUNC_TABLE[self.sm](
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            partial(enumerate_gemm_operators, layout_b=layout_b),
            lambda align: align == 1,  # Only request align1 kernels
            use_3xtf32,
            profile_all_alignments=True,  # To include all align1 kernels
            # TODO(masahi): Invesitigate when fp32 accumulation is needed for gemm
            accumlator_dtype=out_dtype,
        )

        default_kernel_name = DEFAULT_KERNELS[self.sm][(arg0_dtype, out_dtype)]

        if arg0_dtype == "float32":
            default_kernel_name = (
                default_kernel_name[0] if not use_3xtf32 else default_kernel_name[1]
            )

        filtered = list(filter(lambda op: op["name"] == default_kernel_name, ops))
        assert len(filtered) == 1
        op = filtered[0]
        name, opdef = create_gemm_operator_with_epilogue(
            op_type,
            op["tile_description"],
            op["data_type"],
            op["alignment"],
            op["swizzle_functor"],
            batched=batched,
            layout_b=layout_b,
        )
        op.update({"name": name, "opdef": opdef})
        return op

    def select_op(
        self,
        M,
        N,
        K,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        use_3xtf32,
        profile_all_alignments=False,
        find_first_valid=False,
        use_multiprocessing=False,
        layout_b=LayoutType.ColumnMajor,
    ):
        """
        Profile and select the best kernel from candidate kernels.
        See the documentation for the profile method below.
        """
        if (M, N, K) in self.cache:
            op = self.cache[(M, N, K)]
            return op

        # TODO(masahi): CUTLASS alignment check on gemm kernels is too restrictive.
        # See https://github.com/NVIDIA/cutlass/issues/362.
        # When the above issue is resolved, we can remove the alignment check on M below.

        ops = GENERATOR_FUNC_TABLE[self.sm](
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            partial(enumerate_gemm_operators, layout_b=layout_b),
            lambda align: all([dim % align == 0 for dim in [M, N, K]]),
            use_3xtf32,
            profile_all_alignments=profile_all_alignments,
            # TODO(masahi): Invesitigate when fp32 accumulation is needed for gemm
            accumlator_dtype=out_dtype,
        )

        if not find_first_valid:
            self.engine.compile_all(ops, use_multiprocessing)

        for op in ops:
            out = self.engine.evaluate(op, [M, N, K])
            op["runtime"] = out
            if out < float("inf") and find_first_valid:
                self.cache[(M, N, K)] = op
                return op

        op = min(ops, key=lambda i: i["runtime"])
        self.cache[(M, N, K)] = op
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)
        return op

    def profile(
        self,
        op_type,
        M,
        N,
        K,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        use_3xtf32=True,
        profile_all_alignments=False,
        find_first_valid=False,
        use_multiprocessing=False,
        batched=False,
        layout_b=LayoutType.ColumnMajor,
    ):
        """Profile and select the best kernel from candidate kernels.
        If find_first_valid is True, return immediately after the first applicable kernel is found.
        If use_multiprocessing is True, compile all profiler executables in parallel.
        """
        op = self.select_op(
            M,
            N,
            K,
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            use_3xtf32,
            profile_all_alignments=profile_all_alignments,
            find_first_valid=find_first_valid,
            use_multiprocessing=use_multiprocessing,
            layout_b=layout_b,
        )

        name, opdef = create_gemm_operator_with_epilogue(
            op_type,
            op["tile_description"],
            op["data_type"],
            op["alignment"],
            op["swizzle_functor"],
            batched=batched,
            layout_b=layout_b,
        )

        return name, opdef, op["runtime"]
