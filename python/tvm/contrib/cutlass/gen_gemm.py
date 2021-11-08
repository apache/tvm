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
"""Kernel generator and profiler for CUTLASS."""
import logging
import os
import re
import tempfile
import subprocess
import multiprocessing
from .gemm_operation import GemmOperation, EmitGemmInstance
from .gemm_profiler import GemmProfilerEmitter
from .library import (
    EpilogueFunctor,
    SwizzlingFunctor,
    TensorDescription,
    DataTypeTag,
    LayoutType,
    MathInstruction,
    DataType,
    OpcodeClass,
    MathOperation,
    TileDescription,
)

logger = logging.getLogger("cutlass")


def create_gemm_operator(
    layouts,
    tile_descriptions,
    data_type,
    alignment_constraints,
    epilogue_functor=EpilogueFunctor.LinearCombination,
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
                    epilogue_functor,
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

                kernel_emitter = EmitGemmInstance()
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
                ret.append(op_entry)
    return ret


def generate_tensor_op_common(
    math_instructions, alignment_constraints, get_tile_descriptions, batched=False
):
    """Common kernel generator to be used by archtecture specific generators."""
    ops = []
    layouts = [
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    ]
    for math_inst in math_instructions:
        tile_descriptions = get_tile_descriptions(math_inst)
        data_type = [
            math_inst.element_a,
            math_inst.element_b,
            math_inst.element_accumulator,
            math_inst.element_accumulator,
        ]

        out = create_gemm_operator(
            layouts, tile_descriptions, data_type, alignment_constraints, batched=batched
        )

        ops.extend(out)

    return ops


def generate_sm75_tensor_op_1688(out_dtype, batched=False):
    """Generate GEMM kernels for Turing."""
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
        math_instructions, alignment_constraints, get_tile_descriptions, batched
    )


def generate_sm80_tensor_op_16816(out_dtype, batched=False):
    """Generate GEMM kernels for Ampere."""
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

    return generate_tensor_op_common(
        math_instructions, alignment_constraints, get_tile_descriptions, batched
    )


GENERATOR_FUNC_TABLE = {
    75: generate_sm75_tensor_op_1688,
    80: generate_sm80_tensor_op_16816,
}

# TODO(masahi): A sensible way to pick reasonable default kernels
DEFAULT_KERNELS = {
    75: {
        "float16": "cutlass_tensorop_h1688gemm_128x64_32x2_tn_align4",
        "float32": "cutlass_tensorop_s1688gemm_f16_64x64_32x2_tn_align4",
    },
    80: {
        "float16": "cutlass_tensorop_h16816gemm_128x256_32x3_tn_align4",
        "float32": "cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align4",
    },
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
        if args is not None:
            cmd.append(str(args[0]))
            cmd.append(str(args[1]))
            cmd.append(str(args[2]))
            if len(args) > 3:
                cmd.append(str(args[3]))
        try:
            sp = subprocess.run(cmd, capture_output=True, check=True)
            rt = float(sp.stdout)
            logger.info("%s, %f", op_name, rt)
        except subprocess.CalledProcessError:
            rt = -1
        return rt


class CutlassGemmProfiler(object):
    """Profile all candidate kernels and select the best one."""

    def __init__(self, sm, cutlass_path, binary_path):
        assert sm in GENERATOR_FUNC_TABLE and sm in DEFAULT_KERNELS, "sm%d not supported yet." % sm
        self.engine = ProfilerEngine(sm, cutlass_path, binary_path)
        self.sm = sm
        self.cache = {}

    def check_align(self, op_name, M):
        """Filter out kernels that cannot be supported."""
        aligns = re.findall(r"align[1|2|4|8]", op_name)
        assert len(aligns) == 1
        align = int(aligns[0][-1])
        if M % align != 0:
            return False
        return True

    def get_default(self, out_dtype, batched=False):
        """Return the default kernel for the requested architecture.
        For now, the default kernel was picked arbitrary.
        """
        ops = GENERATOR_FUNC_TABLE[self.sm](out_dtype, batched)
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

        ops = GENERATOR_FUNC_TABLE[self.sm](out_dtype, batched)
        ops = list(filter(lambda op: self.check_align(op["name"], M), ops))

        for op in ops:
            op["runtime"] = -1

        if profile_all:
            self.engine.compile_all(ops, use_multiprocessing)

        for op in ops:
            out = self.engine.evaluate(op, [M, N, K])
            op["runtime"] = out
            if out > 0 and profile_all is False:
                break

        valid_ops = filter(lambda op: op["runtime"] > 0, ops)
        output = sorted(valid_ops, key=lambda i: i["runtime"])
        self.cache[(M, N, K)] = output[0]
        return output[0]
