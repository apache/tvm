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
from collections import namedtuple
from .gemm_operation import GemmOperation, EmitGemmInstance, EmitGemmEpilogueInstance
from .gemm_profiler import GemmProfiler
from .compile_engine import CompileEngine
from .library import *


def CreateGemmOperator(
    layouts,
    tile_descriptions,
    data_type,
    alignment_constraints,
    complex_transforms=None,
    epilogue_functor=EpilogueFunctor.LinearCombination,
    swizzling_functor=SwizzlingFunctor.Identity8,
):
    ret = []
    emiter = EmitGemmInstance()
    profiler = GemmProfiler()

    if complex_transforms is None:
        complex_transforms = [
            (ComplexTransform.none, ComplexTransform.none),
        ]

    element_a, element_b, element_c, element_epilogue = data_type
    # by default, only generate the largest tile and largest alignment
    # if manifest.args.kernels == '':
    #  tile_descriptions = [tile_descriptions[0],]
    #  alignment_constraints = [alignment_constraints[0],]

    for layout in layouts:
        for tile_description in tile_descriptions:
            for alignment in alignment_constraints:
                for complex_transform in complex_transforms:

                    alignment_c = min(8, alignment)

                    A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
                    B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
                    C = TensorDescription(element_c, layout[2], alignment_c)

                    op_entry = {}
                    op = GemmOperation(
                        GemmKind.Universal,
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
                        GemmKind.Universal,
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
                        GemmKind.Universal,
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
                        GemmKind.Universal,
                        tile_description.minimum_compute_capability,
                        tile_description,
                        A,
                        B,
                        C,
                        element_epilogue,
                        EpilogueFunctor.LinearCombinationGelu,
                        swizzling_functor,
                    )
                    op_bias_hardswish = GemmOperation(
                        GemmKind.Universal,
                        tile_description.minimum_compute_capability,
                        tile_description,
                        A,
                        B,
                        C,
                        element_epilogue,
                        EpilogueFunctor.LinearCombinationHardswish,
                        swizzling_functor,
                    )
                    op_entry["op"] = op
                    op_entry["name"] = op.procedural_name()
                    emiter = EmitGemmInstance()
                    op_entry["opdef"] = emiter.emit(op)
                    emiter = EmitGemmEpilogueInstance()
                    op_entry["opdef_bias"] = emiter.emit(op_bias)
                    emiter = EmitGemmEpilogueInstance()
                    op_entry["opdef_bias_relu"] = emiter.emit(op_bias_relu)
                    emiter = EmitGemmEpilogueInstance()
                    op_entry["opdef_bias_gelu"] = emiter.emit(op_bias_gelu)
                    emiter = EmitGemmEpilogueInstance()
                    op_entry["opdef_bias_hardswish"] = emiter.emit(op_bias_hardswish)
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
        # (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),
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

    min_cc = 75
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


GENERATOR_FUNC_TABLE = {"GenerateSM75_TensorOp_1688": GenerateSM75_TensorOp_1688}


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
