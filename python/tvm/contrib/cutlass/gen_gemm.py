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
import os
from jinja2 import Template
from .gemm_operation import GemmOperation, EmitGemmInstance
from .library import *
import tempfile
import subprocess


class GemmProfiler(object):
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "profiler_template.cu")) as f:
            self.template = Template(f.read())

    def emit(self, op_name, op_def, dtype_a, dtype_b, dtype_c, ld):
        src = self.template.render(
            OperatorName=op_name,
            OperatorDef=op_def,
            DTypeA=dtype_a,
            DTypeB=dtype_b,
            DTypeC=dtype_c,
            LeadingDim=ld,
        )
        return src


def CreateGemmOperator(
    layouts,
    tile_descriptions,
    data_type,
    alignment_constraints,
    epilogue_functor=EpilogueFunctor.LinearCombination,
    swizzling_functor=SwizzlingFunctor.Identity8,
):
    ret = []
    emiter = EmitGemmInstance()
    profiler = GemmProfiler()

    element_a, element_b, element_c, element_epilogue = data_type

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

                emiter = EmitGemmInstance()
                op_entry["op"] = op
                op_entry["name"] = op.procedural_name()
                op_entry["opdef"] = emiter.emit(op)
                op_entry["opdef_bias"] = emiter.emit(op_bias, no_beta_scaling=True)
                op_entry["opdef_bias_relu"] = emiter.emit(op_bias_relu, no_beta_scaling=True)
                op_entry["opdef_bias_gelu"] = emiter.emit(op_bias_gelu)
                op_entry["src"] = profiler.emit(
                    op.procedural_name(),
                    op_entry["opdef"],
                    DataTypeTag[element_a],
                    DataTypeTag[element_b],
                    DataTypeTag[element_c],
                    op.leading_dim(),
                )
                op_entry["runtime"] = 9999999
                ret.append(op_entry)
    return ret


def GenerateSM75_TensorOp_1688(dtype):
    ops = []
    layouts = [
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    ]

    math_instructions = [
        MathInstruction(
            [16, 8, 8],
            DataType.f16,
            DataType.f16,
            DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
        MathInstruction(
            [16, 8, 8],
            DataType.f16,
            DataType.f16,
            DataType.f16,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
        # TODO: Is the instruction shape correct?
        MathInstruction(
            [16, 8, 8],
            DataType.f32,
            DataType.f32,
            DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
    ]

    min_cc = 80
    max_cc = 1024

    alignment_constraints = [8, 4, 2, 1]

    for math_inst in math_instructions:
        # float32 can and only can use the 32,32,32 MathInstruction.
        if (dtype == "float32" and math_inst.element_a == DataType.f16) or (
            dtype == "float16" and math_inst.element_a == DataType.f32
        ):
            continue

        tile_descriptions = [
            TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 128, 64], 2, [1, 2, 2], math_inst, min_cc, max_cc),
        ]

        data_type = [
            math_inst.element_a,
            math_inst.element_b,
            math_inst.element_accumulator,
            math_inst.element_accumulator,
        ]

        out = CreateGemmOperator(layouts, tile_descriptions, data_type, alignment_constraints)

        # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
        if math_inst.element_a != math_inst.element_accumulator:
            data_type_mixed = [
                math_inst.element_a,
                math_inst.element_b,
                math_inst.element_a,
                math_inst.element_accumulator,
            ]
            out = CreateGemmOperator(
                layouts, tile_descriptions, data_type_mixed, alignment_constraints
            )

        ops.extend(out)
    return ops


def GenerateSM80_TensorOp_16816(dtype):
    assert dtype == "float16"
    ops = []
    layouts = [
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    ]

    math_instructions = [
        MathInstruction(
            [16, 8, 16],
            DataType.f16,
            DataType.f16,
            DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
        MathInstruction(
            [16, 8, 16],
            DataType.f16,
            DataType.f16,
            DataType.f16,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
    ]

    min_cc = 80
    max_cc = 1024
    max_cc_smem_limited = 80

    alignment_constraints = [8, 4, 2]

    for math_inst in math_instructions:
        tile_descriptions = [
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

        data_type = [
            math_inst.element_a,
            math_inst.element_b,
            math_inst.element_accumulator,
            math_inst.element_accumulator,
        ]

        out = CreateGemmOperator(layouts, tile_descriptions, data_type, alignment_constraints)

        # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
        if math_inst.element_a != math_inst.element_accumulator:

            data_type_mixed = [
                math_inst.element_a,
                math_inst.element_b,
                math_inst.element_a,
                math_inst.element_accumulator,
            ]

            out = CreateGemmOperator(
                layouts, tile_descriptions, data_type_mixed, alignment_constraints
            )

        ops.extend(out)
    return ops


GENERATOR_FUNC_TABLE = {
    "GenerateSM75_TensorOp_1688": GenerateSM75_TensorOp_1688,
    "GenerateSM80_TensorOp_16816": GenerateSM80_TensorOp_16816,
}


class CompileEngine(object):
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

    def _compile(self, op_name, src):
        os.makedirs(self.binary_prefix, exist_ok=True)
        opath = os.path.join(self.binary_prefix, op_name)
        if os.path.exists(opath):
            return
        fi = tempfile.NamedTemporaryFile("w", delete=False, suffix=".cu")
        fi.write(src)
        fi.close()
        cmd = self.cmd.format(cflags=self.cflags, src=fi.name, output=opath)
        os.system(cmd)
        os.unlink(fi.name)

    def _execute(self, op_name, args):
        opath = os.path.join(self.binary_prefix, op_name)
        cmd = [opath]
        if args is not None:
            cmd.append(str(args[0]))
            cmd.append(str(args[1]))
            cmd.append(str(args[2]))
            if len(args) > 3:
                cmd.append(str(args[3]))
        sp = subprocess.run(cmd, capture_output=True)
        try:
            print("command: " + str(cmd))
            print("time: " + str(sp.stdout))
            rt = float(sp.stdout)
        except:
            rt = 9999999
        return rt

    def evaluate(self, op_name, src, args=None):
        self._compile(op_name, src)
        return self._execute(op_name, args)


class CutlassGemmProfiler(object):
    def __init__(self, cuda_arch, cutlass_path, binary_path):
        self.engine = CompileEngine(cuda_arch, cutlass_path, binary_path)

    # find out kernels that cannot be supported
    def check_align(self, op_name, M, N, K):
        aligns = re.findall(r"align[1|2|4|8]", op_name)
        assert len(aligns) == 1
        align = int(aligns[0][-1])
        if M % align != 0:
            return False
        else:
            return True

    def profile(self, op_geneators, dtype, M, N, K):
        ops = []
        if isinstance(op_geneators, str):
            op_geneators = [op_geneators]
        for gen in op_geneators:
            ops += GENERATOR_FUNC_TABLE[gen](dtype)
        for op in ops:
            if not self.check_align(op["name"], M, N, K):
                continue
            print(op["name"])
            out = self.engine.evaluate(op["name"], op["src"], [M, N, K])
            op["runtime"] = out
        output = sorted(ops, key=lambda i: i["runtime"])
        return output[0]
