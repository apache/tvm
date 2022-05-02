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
# pylint: disable=invalid-name, dangerous-default-value
"""Conv2d kernel generator and profiler for CUTLASS."""
from functools import partial
from .conv2d_operation import Conv2dOperation, EmitConv2dInstance
from .gen_gemm import CutlassGemmProfiler
from .conv2d_profiler import Conv2dProfilerEmitter
from .gen_tensor_op import ProfilerEngine, GENERATOR_FUNC_TABLE, EPILOGUE_MAP
from .library import (
    DataType,
    EpilogueFunctor,
    SwizzlingFunctor,
    TensorDescription,
    LayoutType,
    ConvKind,
    StrideSupport,
    IteratorAlgorithm,
)


def create_conv2d_operator_with_epilogue(
    conv_kind,
    stride_support,
    op_type,
    tile_description,
    data_type,
    alignment,
    swizzling_functor,
    split_k_slices,
):
    """
    Instantiate a cutlass kernel from the given configuration,
    along with the epilouge functor
    """
    if "residual" in op_type:
        activation_map = {
            "cutlass.conv2d_bias_hardswish": "cutlass::epilogue::thread::HardSwish",
            "cutlass.conv2d_bias_silu": "cutlass::epilogue::thread::SiLu",
            "cutlass.conv2d_bias_sigmoid": "cutlass::epilogue::thread::Sigmoid",
            "cutlass.conv2d_bias_relu": "cutlass::epilogue::thread::ReLu",
            "cutlass.conv2d_bias": "cutlass::epilogue::thread::Identity",
        }
        prefix = op_type[: op_type.find("_residual")]
        activation = activation_map[prefix]
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

    element_a, element_b, element_c, element_epilogue = data_type

    A = TensorDescription(element_a, LayoutType.TensorNHWC, alignment)
    B = TensorDescription(element_b, LayoutType.TensorNHWC, alignment)
    C = TensorDescription(element_c, LayoutType.TensorNHWC, alignment)

    op = Conv2dOperation(
        conv_kind,
        IteratorAlgorithm.Optimized,
        tile_description.minimum_compute_capability,
        tile_description,
        A,
        B,
        C,
        element_epilogue,
        stride_support,
        epilogue,
        swizzling_functor,
        split_k_slices,
    )

    name = op.procedural_name()
    opdef = EmitConv2dInstance().emit(
        op,
        no_beta_scaling=no_beta_scaling,
        residual_block_info=residual_block_info,
        emit_reduction=split_k_slices > 1,
    )

    return name, opdef


def enumerate_conv2d_operators(
    conv_kind,
    stride_support,
    split_k_slices,
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

    if conv_kind == ConvKind.Dgrad and stride_support == StrideSupport.Strided:
        swizzling_functor = SwizzlingFunctor.StridedDgradIdentity1

    for split_k_slice in split_k_slices:
        for tile in tile_descriptions:
            for alignment in alignment_constraints:

                A = TensorDescription(element_a, LayoutType.TensorNHWC, alignment)
                B = TensorDescription(element_b, LayoutType.TensorNHWC, alignment)
                C = TensorDescription(element_c, LayoutType.TensorNHWC, alignment)

                if element_c == DataType.s32 and A.alignment == 1:
                    tile.threadblock_shape[0] = min(tile.threadblock_shape[0], 128)
                    tile.threadblock_shape[1] = min(tile.threadblock_shape[1], 128)

                op = Conv2dOperation(
                    conv_kind,
                    IteratorAlgorithm.Optimized,
                    tile.minimum_compute_capability,
                    tile,
                    A,
                    B,
                    C,
                    element_epilogue,
                    stride_support,
                    EpilogueFunctor.LinearCombination,
                    swizzling_functor,
                    split_k_slice,
                )

                ret.append(
                    {
                        "src": profiler_emitter.emit(
                            kernel_emitter.emit(op, emit_reduction=split_k_slice > 1),
                            op.procedural_name(),
                            element_output=element_c,
                            split_k_slices=split_k_slice,
                        ),
                        "name": op.procedural_name(),
                        "tile_description": tile,
                        "alignment": alignment,
                        "data_type": data_type,
                        "swizzle_functor": swizzling_functor,
                        "split_k_slices": split_k_slice,
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

    def get_default(
        self,
        op_type,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        use_3xtf32,
        conv_kind=ConvKind.Fprop,
        stride=(1, 1),
    ):
        """Return the default kernel for the requested architecture.
        For now, the default kernel was picked arbitrary.
        """
        gemm_profile_result = self.gemm_profiler.get_default(
            op_type, out_dtype, arg0_dtype, arg1_dtype, use_3xtf32
        )
        tile_description = gemm_profile_result["tile_description"]
        alignment = gemm_profile_result["alignment"]
        data_type = gemm_profile_result["data_type"]
        stride_support = StrideSupport.Strided if stride[0] > 1 else StrideSupport.Unity

        if conv_kind == ConvKind.Dgrad and stride_support == StrideSupport.Strided:
            swizzling_functor = SwizzlingFunctor.StridedDgradIdentity1
        else:
            swizzling_functor = SwizzlingFunctor.Identity4

        name, opdef = create_conv2d_operator_with_epilogue(
            conv_kind,
            stride_support,
            op_type,
            tile_description,
            data_type,
            alignment,
            swizzling_functor,
            split_k_slices=1,
        )
        return {"name": name, "opdef": opdef}

    def select_op(
        self,
        d_shape,
        w_shape,
        padding,
        stride,
        dilation,
        out_dtype,
        data_dtype,
        weight_dtype,
        use_3xtf32,
        conv_kind,
        stride_support,
        split_k_slices,
        profile_all_alignments=False,
        find_first_valid=False,
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
            data_dtype,
            weight_dtype,
            partial(enumerate_conv2d_operators, conv_kind, stride_support, split_k_slices),
            lambda align: all([dim % align == 0 for dim in [IC, OC]]),
            use_3xtf32,
            profile_all_alignments,
            # Use fp32 accumulation for wgrad to align with cuDNN
            accumlator_dtype="float32" if conv_kind == ConvKind.Wgrad else out_dtype,
        )

        if not find_first_valid:
            self.engine.compile_all(ops, use_multiprocessing)

        args = (
            "--n=%d --h=%d --w=%d --c=%d --k=%d --r=%d --s=%d --pad_h=%d --pad_w=%d "
            "--stride_h=%d --stride_w=%d --dilation_h=%d --dilation_w=%d"
        ) % workload

        for op in ops:
            out = self.engine.evaluate(op, args.split(" "))
            op["runtime"] = out
            if out < float("inf") and find_first_valid:
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
        data_dtype,
        weight_dtype,
        use_3xtf32=True,
        conv_kind=ConvKind.Fprop,
        split_k_slices=[1],
        profile_all_alignments=False,
        find_first_valid=False,
        use_multiprocessing=False,
    ):
        """Profile and select the best kernel from candidate kernels.
        If find_first_valid is True, return immediately after the first applicable kernel is found.
        If use_multiprocessing is True, compile all profiler executables in parallel.
        """
        # Dgrad requires Unity stride when stride == (1, 1)
        stride_support = (
            StrideSupport.Unity
            if stride[0] == 1 and stride[1] == 1 and conv_kind == ConvKind.Dgrad
            else StrideSupport.Strided
        )

        op = self.select_op(
            d_shape,
            w_shape,
            padding,
            stride,
            dilation,
            out_dtype,
            data_dtype,
            weight_dtype,
            use_3xtf32,
            conv_kind,
            stride_support,
            split_k_slices,
            profile_all_alignments,
            find_first_valid,
            use_multiprocessing,
        )

        name, opdef = create_conv2d_operator_with_epilogue(
            conv_kind,
            stride_support,
            op_type,
            op["tile_description"],
            op["data_type"],
            op["alignment"],
            op["swizzle_functor"],
            op["split_k_slices"],
        )

        return name, opdef, op["runtime"]
