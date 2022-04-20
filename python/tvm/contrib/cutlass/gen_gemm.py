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
from .gemm_operation import GemmOperation, EmitGemmInstance
from .gemm_profiler import GemmProfilerEmitter
from .gen_tensor_op import ProfilerEngine, GENERATOR_FUNC_TABLE, EPILOGUE_MAP
from .library import (
    DataType,
    EpilogueFunctor,
    SwizzlingFunctor,
    TensorDescription,
    DataTypeTag,
    LayoutType,
)


def create_gemm_operator_with_epilogue(
    op_type,
    tile_description,
    data_type,
    alignment,
    swizzling_functor,
    batched=False,
):
    """
    Instantiate a cutlass kernel from the given configuration,
    along with the epilouge functor
    """
    element_a, element_b, element_c, element_epilogue = data_type

    A = TensorDescription(element_a, LayoutType.RowMajor, alignment)
    B = TensorDescription(element_b, LayoutType.ColumnMajor, alignment)
    C = TensorDescription(element_c, LayoutType.RowMajor, alignment)

    if batched:
        swizzling_functor = SwizzlingFunctor.Batched

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
        EmitGemmInstance().emit(op, no_beta_scaling=no_beta_scaling, batched=batched),
    )


def enumerate_gemm_operators(
    tile_descriptions,
    data_type,
    alignment_constraints,
    swizzling_functor=SwizzlingFunctor.Identity8,
):
    """Exhaustively instantiate all kernels from a given configuration."""
    ret = []
    kernel_emitter = EmitGemmInstance()
    profiler_emitter = GemmProfilerEmitter()

    element_a, element_b, element_c, element_epilogue = data_type

    for tile_description in tile_descriptions:
        for alignment in alignment_constraints:
            A = TensorDescription(element_a, LayoutType.RowMajor, alignment)
            B = TensorDescription(element_b, LayoutType.ColumnMajor, alignment)
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
        assert sm in GENERATOR_FUNC_TABLE and sm in DEFAULT_KERNELS, "sm%d not supported yet." % sm
        self.engine = ProfilerEngine(sm, cutlass_path, binary_path)
        self.sm = sm
        self.cache = {}

    def get_default(
        self, op_type, out_dtype, arg0_dtype, arg1_dtype, use_3xtf32=True, batched=False
    ):
        """Return the default kernel for the requested architecture.
        For now, the default kernel was picked arbitrary.
        """
        ops = GENERATOR_FUNC_TABLE[self.sm](
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            enumerate_gemm_operators,
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
            enumerate_gemm_operators,
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
        )

        name, opdef = create_gemm_operator_with_epilogue(
            op_type,
            op["tile_description"],
            op["data_type"],
            op["alignment"],
            op["swizzle_functor"],
            batched=batched,
        )

        return name, opdef, op["runtime"]
