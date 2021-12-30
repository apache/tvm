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
import tempfile
import subprocess
import multiprocessing
from .library import (
    MathInstruction,
    DataType,
    OpcodeClass,
    MathOperation,
    TileDescription,
    EpilogueFunctor,
)

logger = logging.getLogger("cutlass")


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
            math_inst.element_accumulator,
            math_inst.element_accumulator,
        ]

        out = op_creator(tile_descriptions, data_type, alignment_constraints)

        ops.extend(out)

    return ops


def generate_sm75_tensor_op_1688(out_dtype, op_creator):
    """Generate GEMM or Conv2D kernels for Turing."""
    assert out_dtype in ["float32", "float16"]
    math_instructions = {
        "float32": [
            MathInstruction(
                [16, 8, 8],
                DataType.f16,
                DataType.f16,
                DataType.f32,
                OpcodeClass.TensorOp,
                MathOperation.multiply_add,
            )
        ],
        "float16": [
            MathInstruction(
                [16, 8, 8],
                DataType.f16,
                DataType.f16,
                DataType.f16,
                OpcodeClass.TensorOp,
                MathOperation.multiply_add,
            )
        ],
    }[out_dtype]

    alignment_constraints = [8, 4, 2, 1]

    def get_tile_descriptions(math_inst):
        min_cc = 75
        max_cc = 1024
        return [
            TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 128, 64], 2, [1, 2, 2], math_inst, min_cc, max_cc),
        ]

    return generate_tensor_op_common(
        math_instructions, alignment_constraints, get_tile_descriptions, op_creator
    )


def generate_sm80_tensor_op_16816(out_dtype, op_creator):
    """Generate GEMM or Conv2D kernels for Ampere."""
    assert out_dtype in ["float32", "float16"]
    math_instructions = {
        "float32": [
            MathInstruction(
                [16, 8, 16],
                DataType.f16,
                DataType.f16,
                DataType.f32,
                OpcodeClass.TensorOp,
                MathOperation.multiply_add,
            )
        ],
        "float16": [
            MathInstruction(
                [16, 8, 16],
                DataType.f16,
                DataType.f16,
                DataType.f16,
                OpcodeClass.TensorOp,
                MathOperation.multiply_add,
            )
        ],
    }[out_dtype]

    alignment_constraints = [8, 4, 2]

    def get_tile_descriptions(math_inst):
        min_cc = 80
        max_cc = 1024
        max_cc_smem_limited = 80
        return [
            TileDescription([256, 128, 32], 3, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 256, 32], 3, [2, 4, 1], math_inst, min_cc, max_cc),
            TileDescription([256, 64, 32], 4, [4, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 256, 32], 4, [1, 4, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 128, 32], 3, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 128, 32], 4, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 128, 32], 5, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 64, 32], 6, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 128, 32], 6, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 64, 32], 10, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([256, 128, 64], 3, [4, 2, 1], math_inst, min_cc, max_cc_smem_limited),
            TileDescription([128, 256, 64], 3, [2, 4, 1], math_inst, min_cc, max_cc_smem_limited),
            TileDescription([256, 64, 64], 4, [4, 1, 1], math_inst, min_cc, max_cc_smem_limited),
            TileDescription([64, 256, 64], 4, [1, 4, 1], math_inst, min_cc, max_cc_smem_limited),
            TileDescription([128, 128, 64], 4, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 64, 64], 3, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 128, 64], 3, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 64, 64], 5, [2, 2, 1], math_inst, min_cc, max_cc),
        ]

    sm75_kernels = generate_sm75_tensor_op_1688(out_dtype, op_creator)
    sm80_kernels = generate_tensor_op_common(
        math_instructions, alignment_constraints, get_tile_descriptions, op_creator
    )
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
}


class ProfilerEngine:
    """Compile and run a given profiler executable."""

    def __init__(self, cuda_arch, cutlass_path, binary_prefix):
        self.cuda_arch = cuda_arch
        self.binary_prefix = binary_prefix
        self.cutlass = cutlass_path
        self.cflags = "-I{cutlass}/include -I{cutlass}/tools/util/include -O3 -std=c++11".format(
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
        fi = tempfile.NamedTemporaryFile("w", delete=False, suffix=".cu")
        fi.write(op["src"])
        fi.close()
        cmd = self.cmd.format(cflags=self.cflags, src=fi.name, output=opath)
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
        cmd = [opath]
        for arg in args:
            cmd.append(str(arg))
        try:
            sp = subprocess.run(cmd, capture_output=True, check=True)
            rt = float(sp.stdout)
            logger.info("%s, %f", op_name, rt)
        except subprocess.CalledProcessError:
            rt = float("inf")
        return rt
