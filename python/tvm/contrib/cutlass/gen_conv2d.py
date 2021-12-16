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
import re
from .conv2d_operation import Conv2dOperation, EmitConv2dInstance
from .gen_gemm import CutlassGemmProfiler
from .conv2d_profiler import Conv2dProfilerEmitter
from .gen_tensor_op import (
    ProfilerEngine,
    GENERATOR_FUNC_TABLE,
)
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
    profiler_emitter = Conv2dProfilerEmitter()

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

                op_entry["opdef"] = kernel_emitter.emit(op)
                op_entry["op"] = op
                op_entry["src"] = profiler_emitter.emit(op_entry["opdef"], op.procedural_name())
                op_entry["name"] = op.procedural_name()

                # fused ops
                for epilogue, opdef, no_bias_scaling in zip(
                    [
                        EpilogueFunctor.LinearCombinationBias,
                        EpilogueFunctor.LinearCombinationRelu,
                        EpilogueFunctor.LinearCombinationSigmoid,
                    ],
                    ["opdef_bias", "opdef_bias_relu", "opdef_bias_sigmoid"],
                    [True, True, False],
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

                    op_entry[opdef] = kernel_emitter.emit(op, no_bias_scaling)

                ret.append(op_entry)

    return ret


class CutlassConv2DProfiler:
    """Profile all candidate kernels and select the best one."""

    def __init__(self, sm, cutlass_path, binary_path):
        self.gemm_profiler = CutlassGemmProfiler(sm, cutlass_path, binary_path)
        self.sm = sm
        assert sm in GENERATOR_FUNC_TABLE, "sm%d not supported yet." % sm
        self.engine = ProfilerEngine(sm, cutlass_path, binary_path)
        self.cache = {}

    def get_default(self, out_dtype):
        gemm_profile_result = self.gemm_profiler.get_default(out_dtype)
        tile_description = gemm_profile_result["tile_description"]
        alignment = gemm_profile_result["alignment"]
        data_type = gemm_profile_result["data_type"]
        return create_conv2d_operator([tile_description], data_type, [alignment])[0]

    def check_align(self, op_name, C, K):
        """Filter out kernels that cannot be supported."""
        aligns = re.findall(r"align[1|2|4|8]", op_name)
        assert len(aligns) == 1
        align = int(aligns[0][-1])
        return all([dim % align == 0 for dim in [C, K]])

    def profile(
        self,
        d_shape,
        w_shape,
        padding,
        stride,
        dilation,
        out_dtype,
        profile_all=True,
        use_multiprocessing=False,
    ):
        """Profile and select the best kernel from candidate kernels.
        If profile_all is False, return immediately after the first applicable kernel is found.
        If use_multiprocessing is True, compile all profiler executables in parallel.
        """
        N, H, W, IC = d_shape
        OC, R, S, _ = w_shape
        workload = (
            N,
            H,
            W,
            IC,
            OC,
            R,
            S,
            padding[0],
            padding[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
        )

        if workload in self.cache:
            return self.cache[workload]

        ops = GENERATOR_FUNC_TABLE[self.sm](out_dtype, op_creator=create_conv2d_operator)
        ops = list(filter(lambda op: self.check_align(op["name"], IC, OC), ops))

        if profile_all:
            self.engine.compile_all(ops, use_multiprocessing)

        args = (
            "--n=%d --h=%d --w=%d --c=%d --k=%d --r=%d --s=%d --pad_h=%d --pad_w=%d "
            "--stride_h=%d --stride_w=%d --dilation_h=%d --dilation_w=%d"
        ) % workload

        for op in ops:
            out = self.engine.evaluate(op, args.split(" "))
            op["runtime"] = out
            if out < float("inf") and not profile_all:
                self.cache[workload] = op
                return op

        output = min(ops, key=lambda i: i["runtime"])
        self.cache[workload] = output
        return output
