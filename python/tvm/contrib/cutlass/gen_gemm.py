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
from functools import partial
import re
from .gemm_operation import GemmOperation, EmitGemmInstance
from .gemm_profiler import GemmProfilerEmitter
from .gen_tensor_op import (
    ProfilerEngine,
    GENERATOR_FUNC_TABLE,
)
from .library import (
    EpilogueFunctor,
    SwizzlingFunctor,
    TensorDescription,
    DataTypeTag,
    LayoutType,
)


def create_gemm_operator(
    tile_descriptions,
    data_type,
    alignment_constraints,
    swizzling_functor=SwizzlingFunctor.Identity8,
    batched=False,
):
    """Exhaustively instantiate all kernels from a given configuration."""
    ret = []
    kernel_emitter = EmitGemmInstance()
    profiler_emitter = GemmProfilerEmitter()

    element_a, element_b, element_c, element_epilogue = data_type

    if batched:
        swizzling_functor = SwizzlingFunctor.Batched

    layouts = [
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    ]

    for layout in layouts:
        for tile_description in tile_descriptions:
            for alignment in alignment_constraints:
                alignment_c = min(8, alignment)

                A = TensorDescription(element_a, layout[0], alignment)
                B = TensorDescription(element_b, layout[1], alignment)
                C = TensorDescription(element_c, layout[2], alignment_c)

                op_entry = {}
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
                op_bias = GemmOperation(
                    tile_description.minimum_compute_capability,
                    tile_description,
                    A,
                    B,
                    C,
                    element_epilogue,
                    EpilogueFunctor.LinearCombinationBias,
                    swizzling_functor,
                )
                op_bias_relu = GemmOperation(
                    tile_description.minimum_compute_capability,
                    tile_description,
                    A,
                    B,
                    C,
                    element_epilogue,
                    EpilogueFunctor.LinearCombinationRelu,
                    swizzling_functor,
                )
                op_bias_gelu = GemmOperation(
                    tile_description.minimum_compute_capability,
                    tile_description,
                    A,
                    B,
                    C,
                    element_epilogue,
                    EpilogueFunctor.LinearCombinationGelu,
                    swizzling_functor,
                )

                op_entry["op"] = op
                op_entry["name"] = op.procedural_name()
                op_entry["opdef"] = kernel_emitter.emit(op, batched=batched)
                op_entry["opdef_bias"] = kernel_emitter.emit(
                    op_bias, no_beta_scaling=True, batched=batched
                )
                op_entry["opdef_bias_relu"] = kernel_emitter.emit(
                    op_bias_relu, no_beta_scaling=True, batched=batched
                )
                op_entry["opdef_bias_gelu"] = kernel_emitter.emit(op_bias_gelu, batched=batched)
                op_entry["src"] = profiler_emitter.emit(
                    op.procedural_name(),
                    kernel_emitter.emit(op, batched=False),
                    DataTypeTag[element_a],
                    DataTypeTag[element_b],
                    DataTypeTag[element_c],
                    op.leading_dim(),
                )
                op_entry["runtime"] = 9999999
                op_entry["tile_description"] = tile_description
                op_entry["alignment"] = alignment
                op_entry["data_type"] = data_type
                ret.append(op_entry)
    return ret


# TODO(masahi): A sensible way to pick reasonable default kernels
DEFAULT_KERNELS = {
    75: {
        "float16": "cutlass_tensorop_h1688gemm_128x64_32x2_tn_align1",
        "float32": "cutlass_tensorop_s1688gemm_f16_64x64_32x2_tn_align1",
    },
    # align1 variants do not seem to be available for sm80
    80: {
        "float16": "cutlass_tensorop_h1688gemm_128x64_32x2_tn_align1",
        "float32": "cutlass_tensorop_s1688gemm_f16_64x64_32x2_tn_align1",
    },
}


class CutlassGemmProfiler:
    """Profile all candidate kernels and select the best one."""

    def __init__(self, sm, cutlass_path, binary_path):
        assert sm in GENERATOR_FUNC_TABLE and sm in DEFAULT_KERNELS, "sm%d not supported yet." % sm
        self.engine = ProfilerEngine(sm, cutlass_path, binary_path)
        self.sm = sm
        self.cache = {}

    def check_align(self, op_name, M, N, K):
        """Filter out kernels that cannot be supported."""
        aligns = re.findall(r"align[1|2|4|8]", op_name)
        assert len(aligns) == 1
        # The same alignment is used for all axes
        align = int(aligns[0][-1])
        # TODO(masahi): CUTLASS alignment check on gemm kernels is too restrictive.
        # See https://github.com/NVIDIA/cutlass/issues/362.
        # When the above issue is resolved, we can remove the alignment check on M below.
        return all([dim % align == 0 for dim in [M, N, K]])

    def get_default(self, out_dtype, batched=False):
        """Return the default kernel for the requested architecture.
        For now, the default kernel was picked arbitrary.
        """
        ops = GENERATOR_FUNC_TABLE[self.sm](
            out_dtype, op_creator=partial(create_gemm_operator, batched=batched)
        )
        default_kernel_name = DEFAULT_KERNELS[self.sm][out_dtype]
        filtered = list(filter(lambda op: op["name"] == default_kernel_name, ops))
        assert len(filtered) == 1
        return filtered[0]

    def profile(
        self, M, N, K, out_dtype, profile_all=True, use_multiprocessing=False, batched=False
    ):
        """Profile and select the best kernel from candidate kernels.
        If profile_all is False, return immediately after the first applicable kernel is found.
        If use_multiprocessing is True, compile all profiler executables in parallel.
        """
        if (M, N, K) in self.cache:
            return self.cache[(M, N, K)]

        ops = GENERATOR_FUNC_TABLE[self.sm](
            out_dtype, op_creator=partial(create_gemm_operator, batched=batched)
        )
        ops = list(filter(lambda op: self.check_align(op["name"], M, N, K), ops))

        if profile_all:
            self.engine.compile_all(ops, use_multiprocessing)

        for op in ops:
            out = self.engine.evaluate(op, [M, N, K])
            op["runtime"] = out
            if out < float("inf") and not profile_all:
                self.cache[(M, N, K)] = op
                return op

        output = min(ops, key=lambda i: i["runtime"])
        self.cache[(M, N, K)] = output
        return output
