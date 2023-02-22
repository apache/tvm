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
"""Common functions and classes for CUTLASS GEMM and Conv2d geneator."""
import logging
import os
import re
import tempfile
import subprocess
import multiprocessing
import tvm._ffi
from tvm.tir import IntImm
from tvm.runtime import Object
from . import _ffi_api as ffi
from .library import (
    MathInstruction,
    DataType,
    DataTypeTag,
    OpcodeClass,
    MathOperation,
    TileDescription,
    EpilogueFunctor,
)
from .gemm_operation import instantiate_gemm_template
from .conv2d_operation import instantiate_conv2d_template


logger = logging.getLogger("cutlass")


dtype_map = {
    "int8": DataType.s8,
    "uint8": DataType.u8,
    "int32": DataType.s32,
    "float32": DataType.f32,
    "float16": DataType.f16,
}


def generate_tensor_op_common(
    math_instructions, alignment_constraints, get_tile_descriptions, op_creator
):
    """Common kernel generator to be used by archtecture specific generators."""
    ops = []
    for math_inst in math_instructions:
        tile_descriptions = get_tile_descriptions(math_inst)
        data_type = [
            math_inst.element_a,
            math_inst.element_b,
            math_inst.element_c,
            math_inst.element_accumulator,
        ]

        out = op_creator(tile_descriptions, data_type, alignment_constraints)

        ops.extend(out)

    return ops


def generate_sm50_simt(
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    op_creator,
    accumulator_dtype="float32",
):
    """Gemerate GEMM or Conv2D SIMT kernels"""
    # pylint: disable=unused-argument
    min_cc = 50
    max_cc = 1024
    if arg0_dtype == "float32" and arg1_dtype == "float32":
        assert out_dtype == "float32" and accumulator_dtype == "float32"
        math_instructions = [
            MathInstruction(
                [1, 1, 1],
                DataType.f32,
                DataType.f32,
                DataType.f32,
                OpcodeClass.Simt,
                MathOperation.multiply_add,
            ),
        ]
        alignment_constraints = [
            1,
        ]
        tile_descriptions = [
            ([128, 128, 8], 2, [4, 2, 1], min_cc, max_cc),
            ([128, 64, 8], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 128, 8], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 64, 8], 2, [2, 1, 1], min_cc, max_cc),
            ([128, 32, 8], 2, [2, 1, 1], min_cc, max_cc),
            ([32, 128, 8], 2, [1, 2, 1], min_cc, max_cc),
        ]

        def get_tile_descriptions(math_inst):
            return [
                TileDescription(threadblock_shape, stages, warp_count, math_inst, min_cc, max_cc)
                for threadblock_shape, stages, warp_count, min_cc, max_cc in tile_descriptions
            ]

        return generate_tensor_op_common(
            math_instructions, alignment_constraints, get_tile_descriptions, op_creator
        )
    else:
        raise NotImplementedError()


def generate_sm75_tensor_op_1688(
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    op_creator,
    check_align,
    _,
    profile_all_alignments=False,
    accumlator_dtype="float32",
):
    """Generate GEMM or Conv2D kernels for Turing."""
    assert out_dtype in ["float32", "float16", "int32"]
    min_cc = 75
    max_cc = 1024

    if arg0_dtype == "float16" and arg1_dtype == "float16":
        math_instructions = [
            MathInstruction(
                [16, 8, 8],
                DataType.f16,
                DataType.f16,
                dtype_map[out_dtype],
                dtype_map[accumlator_dtype],
                OpcodeClass.TensorOp,
                MathOperation.multiply_add,
            )
        ]
        alignment_constraints = [8, 4, 2, 1]
        tile_descriptions = [
            ([256, 128, 32], 2, [4, 2, 1], min_cc, max_cc),
            ([128, 256, 32], 2, [2, 4, 1], min_cc, max_cc),
            ([128, 128, 32], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 128, 32], 2, [2, 2, 1], min_cc, max_cc),
            ([128, 64, 32], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 64, 32], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 128, 64], 2, [1, 2, 2], min_cc, max_cc),
        ]

    elif "int8" in arg0_dtype and "int8" in arg1_dtype:
        assert out_dtype == "int32"
        math_instructions = [
            MathInstruction(
                [8, 8, 16],
                dtype_map[arg0_dtype],
                dtype_map[arg1_dtype],
                DataType.s32,
                DataType.s32,
                OpcodeClass.TensorOp,
                MathOperation.multiply_add_saturate,
            ),
        ]
        alignment_constraints = [16, 8, 4, 2, 1]
        tile_descriptions = [
            ([256, 128, 64], 2, [4, 2, 1], min_cc, max_cc),
            ([128, 256, 64], 2, [2, 4, 1], min_cc, max_cc),
            ([128, 128, 64], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 256, 64], 2, [1, 4, 1], min_cc, max_cc),
            ([256, 64, 64], 2, [4, 1, 1], min_cc, max_cc),
            ([64, 128, 64], 2, [2, 2, 1], min_cc, max_cc),
            ([128, 64, 64], 2, [2, 2, 1], min_cc, max_cc),
            ([64, 64, 64], 2, [2, 2, 1], min_cc, max_cc),
        ]
    elif arg0_dtype == "float32" and arg1_dtype == "float32" and out_dtype == "float32":
        return generate_sm50_simt(
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            op_creator,
            accumlator_dtype,
        )
    else:
        raise NotImplementedError()

    alignment_constraints = [align for align in alignment_constraints if check_align(align)]
    assert len(alignment_constraints) > 0

    if not profile_all_alignments:
        alignment_constraints = [alignment_constraints[0]]

    def get_tile_descriptions(math_inst):
        return [
            TileDescription(threadblock_shape, stages, warp_count, math_inst, min_cc, max_cc)
            for threadblock_shape, stages, warp_count, min_cc, max_cc in tile_descriptions
        ]

    return generate_tensor_op_common(
        math_instructions, alignment_constraints, get_tile_descriptions, op_creator
    )


def generate_sm80_tensor_op_16816(
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    op_creator,
    check_align,
    use_3xtf32=True,
    profile_all_alignments=False,
    accumlator_dtype="float32",
):
    """Generate GEMM or Conv2D kernels for Ampere."""
    min_cc = 80
    max_cc = 1024
    max_cc_smem_limited = 80

    def get_default_tile_descriptions(block_k_factor):
        return [
            ([256, 128, int(32 * block_k_factor)], 3, [4, 2, 1], min_cc, max_cc),
            ([128, 256, int(32 * block_k_factor)], 3, [2, 4, 1], min_cc, max_cc),
            ([256, 64, int(32 * block_k_factor)], 4, [4, 1, 1], min_cc, max_cc),
            ([64, 256, int(32 * block_k_factor)], 4, [1, 4, 1], min_cc, max_cc),
            ([128, 128, int(32 * block_k_factor)], 3, [2, 2, 1], min_cc, max_cc),
            ([128, 128, int(32 * block_k_factor)], 4, [2, 2, 1], min_cc, max_cc),
            ([128, 128, int(32 * block_k_factor)], 5, [2, 2, 1], min_cc, max_cc),
            ([128, 64, int(32 * block_k_factor)], 6, [2, 2, 1], min_cc, max_cc),
            ([64, 128, int(32 * block_k_factor)], 6, [2, 2, 1], min_cc, max_cc),
            ([64, 64, int(32 * block_k_factor)], 10, [2, 2, 1], min_cc, max_cc),
            ([256, 128, int(64 * block_k_factor)], 3, [4, 2, 1], min_cc, max_cc_smem_limited),
            ([128, 256, int(64 * block_k_factor)], 3, [2, 4, 1], min_cc, max_cc_smem_limited),
            ([256, 64, int(64 * block_k_factor)], 4, [4, 1, 1], min_cc, max_cc_smem_limited),
            ([64, 256, int(64 * block_k_factor)], 4, [1, 4, 1], min_cc, max_cc_smem_limited),
            ([128, 128, int(64 * block_k_factor)], 4, [2, 2, 1], min_cc, max_cc),
            ([128, 64, int(64 * block_k_factor)], 3, [2, 2, 1], min_cc, max_cc),
            ([64, 128, int(64 * block_k_factor)], 3, [2, 2, 1], min_cc, max_cc),
            ([64, 64, int(64 * block_k_factor)], 5, [2, 2, 1], min_cc, max_cc),
        ]

    if arg0_dtype == "float16" and arg1_dtype == "float16":
        math_instructions = [
            MathInstruction(
                [16, 8, 16],
                DataType.f16,
                DataType.f16,
                dtype_map[out_dtype],
                dtype_map[accumlator_dtype],
                OpcodeClass.TensorOp,
                MathOperation.multiply_add,
            )
        ]
        alignment_constraints = [8, 4, 2]
        tile_descriptions = get_default_tile_descriptions(1)
    elif arg0_dtype == "float32" and arg1_dtype == "float32":
        math_instructions = [
            MathInstruction(
                [16, 8, 8],
                DataType.f32,
                DataType.f32,
                DataType.f32,
                DataType.f32,
                OpcodeClass.TensorOp,
                MathOperation.multiply_add_fast_f32 if use_3xtf32 else MathOperation.multiply_add,
            ),
        ]
        alignment_constraints = [4, 2, 1]

        if use_3xtf32:
            # tf32
            tile_descriptions = [
                ([128, 128, 16], 4, [4, 2, 1], min_cc, max_cc),
                ([128, 128, 16], 3, [4, 2, 1], min_cc, max_cc),
                ([256, 64, 16], 3, [4, 2, 1], min_cc, max_cc),
                ([64, 256, 16], 3, [2, 4, 1], min_cc, max_cc),
                ([128, 64, 16], 4, [2, 2, 1], min_cc, max_cc),
                ([64, 128, 16], 4, [2, 2, 1], min_cc, max_cc),
                ([64, 64, 16], 3, [2, 2, 1], min_cc, max_cc),
                ([128, 128, 32], 3, [4, 2, 1], min_cc, max_cc),
                ([256, 64, 32], 3, [4, 2, 1], min_cc, max_cc_smem_limited),
                ([64, 256, 32], 3, [2, 4, 1], min_cc, max_cc_smem_limited),
                ([128, 64, 32], 3, [2, 2, 1], min_cc, max_cc),
                ([64, 128, 32], 3, [2, 2, 1], min_cc, max_cc),
                ([64, 64, 32], 3, [2, 2, 1], min_cc, max_cc),
            ]
        else:
            tile_descriptions = get_default_tile_descriptions(0.5)
    else:
        assert out_dtype == "int32"
        math_instructions = [
            MathInstruction(
                [16, 8, 32],
                dtype_map[arg0_dtype],
                dtype_map[arg1_dtype],
                DataType.s32,
                DataType.s32,
                OpcodeClass.TensorOp,
                MathOperation.multiply_add_saturate,
            ),
        ]
        alignment_constraints = [16, 8, 4]
        tile_descriptions = get_default_tile_descriptions(2)

    def get_tile_descriptions(math_inst):
        return [
            TileDescription(threadblock_shape, stages, warp_count, math_inst, min_cc, max_cc)
            for threadblock_shape, stages, warp_count, min_cc, max_cc in tile_descriptions
        ]

    alignment_constraints = [align for align in alignment_constraints if check_align(align)]

    if len(alignment_constraints) > 0 and not profile_all_alignments:
        alignment_constraints = [alignment_constraints[0]]

    if arg0_dtype != "float32" and arg1_dtype != "float32":
        sm75_kernels = generate_sm75_tensor_op_1688(
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            op_creator,
            check_align,
            False,
            profile_all_alignments,
            accumlator_dtype=accumlator_dtype,
        )
    else:
        # TF32 (float32 + float32 case) is only supported on sm80
        sm75_kernels = []

    if len(alignment_constraints) > 0:
        sm80_kernels = generate_tensor_op_common(
            math_instructions, alignment_constraints, get_tile_descriptions, op_creator
        )
    else:
        sm80_kernels = []

    # TODO(masahi): For int8 kernels, The CUTLASS generator modifies the output tensor alignment
    # after ops are created. Revisit how important this modification is.
    # for op in operations:
    #     if op.tile_description.threadblock_shape[1] >= 128:
    #         op.C.alignment = 16
    #     else:
    #         op.C.alignment = 8

    return sm75_kernels + sm80_kernels


GENERATOR_FUNC_TABLE = {
    75: generate_sm75_tensor_op_1688,
    80: generate_sm80_tensor_op_16816,
}


# (Epilogue functor name, no_beta_scaling)
EPILOGUE_MAP = {
    "cutlass.dense": (EpilogueFunctor.LinearCombination, False),
    "cutlass.dense_bias": (EpilogueFunctor.LinearCombinationBias, True),
    "cutlass.dense_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
    "cutlass.dense_bias_gelu_fp16": (EpilogueFunctor.LinearCombinationGelu, False),
    "cutlass.dense_bias_gelu_fp32": (EpilogueFunctor.LinearCombinationGelu, False),
    "cutlass.batch_matmul": (EpilogueFunctor.LinearCombination, False),
    "cutlass.conv2d_bias_hardswish": (EpilogueFunctor.LinearCombinationHardSwish, False),
    "cutlass.conv2d_bias_silu": (EpilogueFunctor.LinearCombinationSilu, False),
    "cutlass.conv2d_bias_sigmoid": (EpilogueFunctor.LinearCombinationSigmoid, False),
    "cutlass.conv2d_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
    "cutlass.conv2d_bias": (EpilogueFunctor.LinearCombinationBias, True),
    "cutlass.conv2d": (EpilogueFunctor.LinearCombination, False),
    "cutlass.conv2d_transpose": (EpilogueFunctor.LinearCombination, False),
    "cutlass.conv2d_backward_weight": (EpilogueFunctor.LinearCombination, False),
}


class ProfilerEngine:
    """Compile and run a given profiler executable."""

    def __init__(self, cuda_arch, cutlass_path, binary_prefix):
        self.cuda_arch = cuda_arch
        self.binary_prefix = binary_prefix
        self.cutlass = cutlass_path
        self.cflags = "-I{cutlass}/include -I{cutlass}/tools/util/include -O3 -std=c++17".format(
            cutlass=cutlass_path
        )
        self.cflags += " -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
        self.cflags += " -gencode=arch=compute_{arch},code=[sm_{arch},compute_{arch}]".format(
            arch=cuda_arch
        )
        self.cflags += " -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing"
        self.cmd = "nvcc {cflags} {src} -o {output}"

    def _compile(self, op):
        os.makedirs(self.binary_prefix, exist_ok=True)
        opath = os.path.join(self.binary_prefix, op["name"])
        if os.path.exists(opath):
            return
        fi = tempfile.NamedTemporaryFile("w", delete=False, prefix=self.binary_prefix, suffix=".cu")
        fi.write(op["src"])
        fi.close()
        cmd = self.cmd.format(cflags=self.cflags, src=fi.name, output=opath)
        logger.info("invoking compilation %s", cmd)
        os.system(cmd)
        os.unlink(fi.name)

    def compile_all(self, ops, use_multiprocessing=False):
        """Compile all profiler executables."""
        if use_multiprocessing:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            pool.map(self._compile, ops)
        else:
            for op in ops:
                self._compile(op)

    def evaluate(self, op, args):
        """Run the profiler executable corresponding to op_name with args."""
        op_name = op["name"]
        opath = os.path.join(self.binary_prefix, op_name)
        if not os.path.exists(opath):
            self._compile(op)
        if not os.path.exists(opath):
            # Bail out if compilation fails for a whatever reason (e.g. static assert failure)
            return float("inf")
        cmd = [opath]
        for arg in args:
            cmd.append(str(arg))
        try:
            logger.info("invoking evaluation %s", cmd)
            sp = subprocess.run(cmd, capture_output=True, check=True)
            rt = float(sp.stdout)
            if rt == 0.0:
                # This seems to happen with split-k using invalid split-k-slices
                rt = float("inf")
            logger.info("%s, %f", op_name, rt)
        except subprocess.CalledProcessError:
            rt = float("inf")
        return rt


class CodegenResult(Object):
    """The holder for the generated code and required headers."""

    def __init__(self, code, headers):
        self.__init_handle_by_constructor__(ffi.CodegenResult, code, headers)


@tvm._ffi.register_func("contrib.cutlass.instantiate_template")
def instantiate_template(func_name, annotations, func_args):
    """Return CUTLASS host code based on a template and the provided annotations.

    Parameters
    ----------
    func_name: str
        A string to identify the type of the kernel (dense/matmul, batched_matmul, or conv2d).

    annotations: container.Map
        Key and value pairs annotated during kernel selection.

    func_args: list
        Names of the function arguments.

    Returns
    -------
    codegen_result : CodegenResult
        Generated CUTLASS host code and required header-file names.
    """
    attrs = {}

    for k in ["lda", "ldb", "ldc", "cutlass_op_def", "cutlass_op_name", "op_type"]:
        if k in annotations:
            attrs[k] = annotations[k]

    arg0_shape = annotations["arg0_shape"]
    arg1_shape = annotations["arg1_shape"]
    attrs["ElementInputA"] = DataTypeTag[dtype_map[annotations["arg0_dtype"]]]
    attrs["ElementInputB"] = DataTypeTag[dtype_map[annotations["arg1_dtype"]]]
    attrs["ElementOutput"] = DataTypeTag[dtype_map[annotations["ret_dtype"]]]

    headers = []

    if "relu" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_bias_relu.h")
    elif "gelu" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_gelu.h")
    elif "sigmoid" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_sigmoid.h")
    elif "silu" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_silu.h")
    elif "hardswish" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_hardswish.h")
    else:
        headers.append("cutlass/epilogue/thread/linear_combination.h")

    if "residual" in func_name:
        headers.append("cutlass/epilogue/thread/linear_combination_residual_block.h")

    def get_dim(shape_annot, var_name, axis_idx, batched_offset=0):
        if isinstance(shape_annot, IntImm):
            return str(int(shape_annot))
        return "{}->shape[{}]".format(var_name, batched_offset + axis_idx)

    def get_batch_stride(stride_annot, arg0_idx, arg1_idx, arg0_axis_idx, arg1_axis_idx):
        if isinstance(stride_annot, IntImm):
            return str(int(stride_annot))
        dim1 = func_args[arg0_idx] + "->shape[{}]".format(arg0_axis_idx)
        dim2 = func_args[arg1_idx] + "->shape[{}]".format(arg1_axis_idx)
        return dim1 + " * " + dim2

    if "dense" in func_name or "matmul" in func_name:
        batched = "batch_matmul" in func_name
        batched_offset = 1 if batched else 0
        attrs["K"] = str(int(arg0_shape[batched_offset + 1]))
        attrs["M"] = get_dim(arg0_shape[batched_offset], func_args[0], 0, batched_offset)

        if annotations["ldb"] == "N":
            attrs["N"] = get_dim(arg1_shape[batched_offset + 1], func_args[1], 1, batched_offset)
        else:
            attrs["N"] = get_dim(arg1_shape[batched_offset], func_args[1], 0, batched_offset)

        if batched:
            headers.append("cutlass/gemm/device/gemm_batched.h")
            attrs["batch"] = get_dim(arg0_shape[0], func_args[0], 0)
            attrs["batch_stride_A"] = get_batch_stride(annotations["batch_stride_A"], 0, 0, 1, 2)
            attrs["batch_stride_B"] = get_batch_stride(annotations["batch_stride_B"], 1, 1, 1, 2)

            if annotations["ldb"] == "N":
                attrs["batch_stride_C"] = get_batch_stride(
                    annotations["batch_stride_C"], 0, 1, 1, 2
                )
            else:
                attrs["batch_stride_C"] = get_batch_stride(
                    annotations["batch_stride_C"], 0, 1, 1, 1
                )
        else:
            headers.append("cutlass/gemm/device/gemm.h")

        code = instantiate_gemm_template(attrs, func_args)
        return CodegenResult(code, headers)

    elif "conv2d" in func_name:
        activation_shape = arg0_shape
        weight_shape = arg1_shape
        output_shape = annotations["ret_shape"]
        activation_var = func_args[0]

        if "conv2d_transpose" in func_name:
            headers.append("cutlass/conv/kernel/default_conv2d_dgrad.h")
            activation_shape = output_shape
            output_shape = arg0_shape
        elif "backward" in func_name:
            headers.append("cutlass/conv/kernel/default_conv2d_wgrad.h")
            activation_shape = arg1_shape
            weight_shape = output_shape
            output_shape = arg0_shape
        elif "residual" in func_name:
            headers.append("cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h")
        else:
            headers.append("cutlass/conv/kernel/default_conv2d_fprop.h")

        headers.append("cutlass/conv/device/implicit_gemm_convolution.h")

        op_name = attrs["cutlass_op_name"]

        if "splitk" in op_name:
            headers += [
                "cutlass/reduction/device/reduce_split_k.h",
                "cutlass/reduction/thread/reduction_operators.h",
            ]

        attrs["N"] = get_dim(activation_shape[0], activation_var, 0)
        attrs["H"] = get_dim(activation_shape[1], activation_var, 1)
        attrs["W"] = get_dim(activation_shape[2], activation_var, 2)
        attrs["C"] = str(int(activation_shape[3]))
        attrs["P"] = get_dim(output_shape[1], "out0", 1)
        attrs["Q"] = get_dim(output_shape[2], "out0", 2)
        attrs["K"] = str(int(output_shape[3]))
        attrs["R"] = str(int(weight_shape[1]))
        attrs["S"] = str(int(weight_shape[2]))
        attrs["pad_h"] = str(int(annotations["padding"][0]))
        attrs["pad_w"] = str(int(annotations["padding"][1]))
        attrs["stride_h"] = str(int(annotations["strides"][0]))
        attrs["stride_w"] = str(int(annotations["strides"][1]))
        attrs["dilation_h"] = str(int(annotations["dilation"][0]))
        attrs["dilation_w"] = str(int(annotations["dilation"][1]))

        if "splitk" in op_name:
            attrs["split_k_mode"] = "kParallel"
            attrs["split_k_slices"] = str(re.search(r"splitk(\d+)", op_name).group(1))
        else:
            attrs["split_k_mode"] = "kSerial"
            attrs["split_k_slices"] = "1"

        code = instantiate_conv2d_template(attrs, func_args)
        return CodegenResult(code, headers)

    raise ValueError("Do not have a template for {}".format(func_name))
