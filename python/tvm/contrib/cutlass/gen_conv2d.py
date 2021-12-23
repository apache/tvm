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
from .gen_tensor_op import ProfilerEngine, GENERATOR_FUNC_TABLE, EPILOGUE_MAP
from .library import (
    EpilogueFunctor,
    SwizzlingFunctor,
    TensorDescription,
    LayoutType,
    ConvKind,
    StrideSupport,
    IteratorAlgorithm,
)


def create_conv2d_operator_with_epilogue(
    op_type, tile_description, data_type, alignment, swizzling_functor
):
    """
    Instantiate a cutlass kernel from the given configuration,
    along with the epilouge functor
    """
    epilogue, no_beta_scaling = EPILOGUE_MAP[op_type]

    element_a, element_b, element_c, element_epilogue = data_type

    A = TensorDescription(element_a, LayoutType.TensorNHWC, alignment)
    B = TensorDescription(element_b, LayoutType.TensorNHWC, alignment)
    C = TensorDescription(element_c, LayoutType.TensorNHWC, alignment)

    op = Conv2dOperation(
        ConvKind.Fprop,
        IteratorAlgorithm.Optimized,
        tile_description.minimum_compute_capability,
        tile_description,
        A,
        B,
        C,
        element_epilogue,
        StrideSupport.Strided,
        epilogue,
        swizzling_functor,
    )

    name = op.procedural_name()
    opdef = EmitConv2dInstance().emit(op, no_beta_scaling=no_beta_scaling)

    return name, opdef


def enumerate_conv2d_operators(
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

    for tile in tile_descriptions:
        for alignment in alignment_constraints:

            A = TensorDescription(element_a, LayoutType.TensorNHWC, alignment)
            B = TensorDescription(element_b, LayoutType.TensorNHWC, alignment)
            C = TensorDescription(element_c, LayoutType.TensorNHWC, alignment)

            op = Conv2dOperation(
                ConvKind.Fprop,
                IteratorAlgorithm.Optimized,
                tile.minimum_compute_capability,
                tile,
                A,
                B,
                C,
                element_epilogue,
                StrideSupport.Strided,
                EpilogueFunctor.LinearCombination,
                swizzling_functor,
            )

            ret.append(
                {
                    "src": profiler_emitter.emit(kernel_emitter.emit(op), op.procedural_name()),
                    "name": op.procedural_name(),
                    "tile_description": tile,
                    "alignment": alignment,
                    "data_type": data_type,
                    "swizzle_functor": swizzling_functor,
                }
            )

    return ret


class CutlassConv2DProfiler:
    """Profile all candidate kernels and select the best one."""

    def __init__(self, sm, cutlass_path, binary_path):
        self.gemm_profiler = CutlassGemmProfiler(sm, cutlass_path, binary_path)
        self.sm = sm
        assert sm in GENERATOR_FUNC_TABLE, "sm%d not supported yet." % sm
        self.engine = ProfilerEngine(sm, cutlass_path, binary_path)
        self.cache = {}

    def get_default(self, op_type, out_dtype):
        gemm_profile_result = self.gemm_profiler.get_default(op_type, out_dtype)
        tile_description = gemm_profile_result["tile_description"]
        alignment = gemm_profile_result["alignment"]
        data_type = gemm_profile_result["data_type"]
        name, opdef = create_conv2d_operator_with_epilogue(
            op_type, tile_description, data_type, alignment, SwizzlingFunctor.Identity4
        )
        return {"name": name, "opdef": opdef}

    def check_align(self, op_name, C, K):
        """Filter out kernels that cannot be supported."""
        aligns = re.findall(r"align[1|2|4|8]", op_name)
        assert len(aligns) == 1
        align = int(aligns[0][-1])
        return all([dim % align == 0 for dim in [C, K]])

    def select_op(
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
        """
        Profile and select the best kernel from candidate kernels.
        See the documentation for the profile method below.
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

        ops = GENERATOR_FUNC_TABLE[self.sm](
            out_dtype,
            op_creator=enumerate_conv2d_operators,
        )
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

        op = min(ops, key=lambda i: i["runtime"])
        self.cache[workload] = op
        return op

    def profile(
        self,
        op_type,
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
        op = self.select_op(
            d_shape,
            w_shape,
            padding,
            stride,
            dilation,
            out_dtype,
            profile_all=profile_all,
            use_multiprocessing=use_multiprocessing,
        )

        name, opdef = create_conv2d_operator_with_epilogue(
            op_type, op["tile_description"], op["data_type"], op["alignment"], op["swizzle_functor"]
        )

        return name, opdef, op["runtime"]
