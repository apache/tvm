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
#
# \file generator.py
#
# \brief Generates the CUTLASS Library's instances
#

import enum
import os.path
import shutil
import functools
import operator

from .library import *


###################################################################################################
#
# Data structure modeling a GEMM operation
#
###################################################################################################

#
class GemmOperation:
    #
    def __init__(
        self,
        gemm_kind,
        arch,
        tile_description,
        A,
        B,
        C,
        element_epilogue,
        epilogue_functor=EpilogueFunctor.LinearCombination,
        swizzling_functor=SwizzlingFunctor.Identity8,
    ):

        self.operation_kind = OperationKind.Gemm
        self.arch = arch
        self.tile_description = tile_description
        self.gemm_kind = gemm_kind
        self.A = A
        self.B = B
        self.C = C
        self.element_epilogue = element_epilogue
        self.epilogue_functor = epilogue_functor
        self.swizzling_functor = swizzling_functor

    #
    def accumulator_type(self):
        return self.tile_description.math_instruction.element_accumulator

    #
    def short_math_name(self):
        return ShortDataTypeNames[self.accumulator_type()]

    #
    def core_name(self):
        """ The basic operation kind is prefixed with a letter indicating the accumulation type. """

        inst_shape = ""
        inst_operation = ""
        intermediate_type = ""

        if (
            self.tile_description.math_instruction.opcode_class == OpcodeClass.TensorOp
            or self.tile_description.math_instruction.opcode_class == OpcodeClass.WmmaTensorOp
        ):

            math_op = self.tile_description.math_instruction.math_operation
            math_op_string = ""

            inst_shape = "%d%d%d" % tuple(self.tile_description.math_instruction.instruction_shape)
            inst_shape += math_op_string

            if (
                self.tile_description.math_instruction.element_a != self.A.element
                and self.tile_description.math_instruction.element_a
                != self.tile_description.math_instruction.element_accumulator
            ):
                intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]

        return "%s%s%s%s" % (
            self.short_math_name(),
            inst_shape,
            intermediate_type,
            GemmKindNames[self.gemm_kind],
        )

    #
    def extended_name(self):
        """ Append data types if they differ from compute type. """
        if (
            self.C.element != self.tile_description.math_instruction.element_accumulator
            and self.A.element != self.tile_description.math_instruction.element_accumulator
        ):
            extended_name = "${element_c}_${core_name}_${element_a}"
        elif (
            self.C.element == self.tile_description.math_instruction.element_accumulator
            and self.A.element != self.tile_description.math_instruction.element_accumulator
        ):
            extended_name = "${core_name}_${element_a}"
        else:
            extended_name = "${core_name}"

        extended_name = SubstituteTemplate(
            extended_name,
            {
                "element_a": DataTypeNames[self.A.element],
                "element_c": DataTypeNames[self.C.element],
                "core_name": self.core_name(),
            },
        )

        return extended_name

    #
    def layout_name(self):
        return "%s%s" % (ShortLayoutTypeNames[self.A.layout], ShortLayoutTypeNames[self.B.layout])

    #
    def procedural_name(self):
        """ The full procedural name indicates architecture, extended name, tile size, and layout. """
        threadblock = self.tile_description.procedural_name()

        opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]

        alignment = max([self.A.alignment, self.B.alignment, self.C.alignment])

        return SubstituteTemplate(
            "cutlass_${opcode_class}_${extended_name}_${threadblock}_${layout}_align${alignment}",
            {
                "opcode_class": opcode_class_name,
                "extended_name": self.extended_name(),
                "threadblock": threadblock,
                "layout": self.layout_name(),
                "alignment": "%d" % self.A.alignment,
            },
        )

    #
    def leading_dim(self):
        """ lda, ldb, ldc, according to the leading dimension. """
        if self.A.layout == LayoutType.RowMajor:
            lda = "K"
        elif self.A.layout == LayoutType.ColumnMajor:
            lda = "M"
        else:
            ValueError("The layout of A is not implemented.")

        if self.B.layout == LayoutType.RowMajor:
            ldb = "N"
        elif self.B.layout == LayoutType.ColumnMajor:
            ldb = "K"
        else:
            ValueError("The layout of B is not implemented.")

        if self.C.layout == LayoutType.RowMajor:
            ldc = "N"
        elif self.C.layout == LayoutType.ColumnMajor:
            ldc = "M"
        else:
            ValueError("The layout of B is not implemented.")

        return SubstituteTemplate(
            "int lda = ${lda_val};\n\tint ldb = ${ldb_val};\n\tint ldc = ${ldc_val};\n",
            {"lda_val": lda, "ldb_val": ldb, "ldc_val": ldc,},
        )

    #
    def configuration_name(self):
        """ The full procedural name indicates architecture, extended name, tile size, and layout. """
        return self.procedural_name()


complex_transform_tag = "cutlass::ComplexTransform::kNone"

###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

#
class EmitGemmInstance:
    """ Responsible for emitting a CUTLASS template definition"""

    def __init__(self):
        self.gemm_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::Gemm<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    false,
    ${math_operation}
    ${residual}
  >;
"""

    def emit(self, operation):

        warp_shape = [
            operation.tile_description.threadblock_shape[idx]
            // operation.tile_description.warp_count[idx]
            for idx in range(3)
        ]
        epilogue_vector_length = int(
            min(operation.C.alignment * DataTypeSize[operation.C.element], 128)
            / DataTypeSize[operation.C.element]
        )
        residual = ""

        values = {
            "operation_name": operation.procedural_name(),
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[operation.A.layout],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[operation.B.layout],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[operation.C.layout],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "opcode_class": OpcodeClassTag[
                operation.tile_description.math_instruction.opcode_class
            ],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(
                operation.tile_description.math_instruction.instruction_shape[0]
            ),
            "instruction_shape_n": str(
                operation.tile_description.math_instruction.instruction_shape[1]
            ),
            "instruction_shape_k": str(
                operation.tile_description.math_instruction.instruction_shape[2]
            ),
            "epilogue_vector_length": str(epilogue_vector_length),
            "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
            "epilogue_functor": EpilogueFunctorTag[operation.epilogue_functor],
            "swizzling_functor": SwizzlingFunctorTag[operation.swizzling_functor],
            "stages": str(operation.tile_description.stages),
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
            "transform_a": complex_transform_tag,
            "transform_b": complex_transform_tag,
            "math_operation": MathOperationTag[
                operation.tile_description.math_instruction.math_operation
            ],
            "residual": residual,
        }
        return SubstituteTemplate(self.gemm_template, values)


###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

#
class EmitGemmEpilogueInstance:
    """ Responsible for emitting a CUTLASS template definition with epilogue scaling"""

    def __init__(self):
        self.gemm_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::Gemm<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue},
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    false,
    ${math_operation}
    ${residual}
  >;
"""
        self.gemm_gelu_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::Gemm<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    false,
    ${math_operation}
    ${residual}
  >;
"""

    def emit(self, operation):

        warp_shape = [
            operation.tile_description.threadblock_shape[idx]
            // operation.tile_description.warp_count[idx]
            for idx in range(3)
        ]
        epilogue_vector_length = int(
            min(operation.C.alignment * DataTypeSize[operation.C.element], 128)
            / DataTypeSize[operation.C.element]
        )

        residual = ""

        values = {
            "operation_name": operation.procedural_name(),
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[operation.A.layout],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[operation.B.layout],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[operation.C.layout],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "opcode_class": OpcodeClassTag[
                operation.tile_description.math_instruction.opcode_class
            ],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(
                operation.tile_description.math_instruction.instruction_shape[0]
            ),
            "instruction_shape_n": str(
                operation.tile_description.math_instruction.instruction_shape[1]
            ),
            "instruction_shape_k": str(
                operation.tile_description.math_instruction.instruction_shape[2]
            ),
            "epilogue_vector_length": str(epilogue_vector_length),
            "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
            "epilogue_functor": EpilogueFunctorTag[operation.epilogue_functor],
            "swizzling_functor": SwizzlingFunctorTag[operation.swizzling_functor],
            "stages": str(operation.tile_description.stages),
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
            "transform_a": complex_transform_tag,
            "transform_b": complex_transform_tag,
            "math_operation": MathOperationTag[
                operation.tile_description.math_instruction.math_operation
            ],
            "residual": residual,
        }

        if values["epilogue_functor"] == EpilogueFunctorTag[EpilogueFunctor.LinearCombinationGelu]:
            template = self.gemm_gelu_template
        else:
            template = self.gemm_template

        return SubstituteTemplate(template, values)


###################################################################################################


#
class EmitGemmUniversalInstance:
    """ Responsible for emitting a CUTLASS template definition"""

    def __init__(self):
        self.gemm_template = """
// Gemm operator ${operation_name}
using ${operation_name}_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},    // transposed B operand
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},    // transposed A operand
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name} :
  public ${operation_name}_base { };
"""
        self.gemm_template_interleaved = """
// Gemm operator ${operation_name}
using ${operation_name}_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name} :
  public ${operation_name}_base { };
"""

    def emit(self, operation):

        threadblock_shape = operation.tile_description.threadblock_shape
        warp_count = operation.tile_description.warp_count

        warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

        epilogue_vector_length = int(
            min(operation.C.alignment * DataTypeSize[operation.C.element], 128)
            / DataTypeSize[operation.C.element]
        )

        transpose_layouts = {
            LayoutType.ColumnMajor: LayoutType.RowMajor,
            LayoutType.RowMajor: LayoutType.ColumnMajor,
        }

        if (
            operation.A.layout in transpose_layouts.keys()
            and operation.B.layout in transpose_layouts.keys()
            and operation.C.layout in transpose_layouts.keys()
        ):

            instance_layout_A = transpose_layouts[operation.A.layout]
            instance_layout_B = transpose_layouts[operation.B.layout]
            instance_layout_C = transpose_layouts[operation.C.layout]

            gemm_template = self.gemm_template
        else:
            instance_layout_A, instance_layout_B, instance_layout_C = (
                operation.A.layout,
                operation.B.layout,
                operation.C.layout,
            )

            gemm_template = self.gemm_template_interleaved
        #

        values = {
            "operation_name": operation.procedural_name(),
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[instance_layout_A],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[instance_layout_B],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[instance_layout_C],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "opcode_class": OpcodeClassTag[
                operation.tile_description.math_instruction.opcode_class
            ],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(
                operation.tile_description.math_instruction.instruction_shape[0]
            ),
            "instruction_shape_n": str(
                operation.tile_description.math_instruction.instruction_shape[1]
            ),
            "instruction_shape_k": str(
                operation.tile_description.math_instruction.instruction_shape[2]
            ),
            "epilogue_vector_length": str(epilogue_vector_length),
            "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
            "epilogue_functor": EpilogueFunctorTag[operation.epilogue_functor],
            "swizzling_functor": SwizzlingFunctorTag[operation.swizzling_functor],
            "stages": str(operation.tile_description.stages),
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
            "transform_a": complex_transform_tag,
            "transform_b": complex_transform_tag,
            "math_operation": MathOperationTag[
                operation.tile_description.math_instruction.math_operation
            ],
        }

        return SubstituteTemplate(gemm_template, values)


###################################################################################################
#
# Emitters functions for all targets
#
###################################################################################################


class EmitGemmConfigurationLibrary:
    def __init__(self, operation_path, configuration_name):
        self.configuration_name = configuration_name
        self.configuration_path = os.path.join(
            operation_path, "%s.cu" % configuration_name
        ).replace("\\", "/")

        self.instance_emitter = {
            GemmKind.Gemm: EmitGemmInstance,
            GemmKind.Universal: EmitGemmUniversalInstance,
        }

        self.gemm_kind_wrappers = {
            GemmKind.Gemm: "GemmOperation",
            GemmKind.Universal: "GemmUniversalOperation",
        }

        self.wmma_guard_start = "#if defined(CUTLASS_ARCH_WMMA_SM${sm_number}_ENABLED)"

        self.instance_template = {
            GemmKind.Gemm: """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<Operation_${operation_name}>("${operation_name}"));
${compile_guard_end}
""",
            GemmKind.Universal: """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
      cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>
    >("${operation_name}"));
${compile_guard_end}
""",
        }

        self.header_template = """
/*
  Generated by gemm_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "gemm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

        self.initialize_function_template = """

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_${configuration_name}(Manifest &manifest) {

"""
        self.epilogue_template = """

}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

    def __enter__(self):
        self.configuration_file = open(self.configuration_path, "w")
        self.configuration_file.write(self.header_template)

        self.instance_definitions = []
        self.instance_wrappers = []

        self.operations = []
        return self

    def emit(self, operation):
        emitter = self.instance_emitter[operation.gemm_kind]()

        self.operations.append(operation)

        self.instance_definitions.append(emitter.emit(operation))

        self.instance_wrappers.append(
            SubstituteTemplate(
                self.instance_template[operation.gemm_kind],
                {
                    "configuration_name": self.configuration_name,
                    "operation_name": operation.procedural_name(),
                    "gemm_kind": self.gemm_kind_wrappers[operation.gemm_kind],
                    "compile_guard_start": SubstituteTemplate(
                        self.wmma_guard_start, {"sm_number": str(operation.arch)}
                    )
                    if operation.tile_description.math_instruction.opcode_class
                    == OpcodeClass.WmmaTensorOp
                    else "",
                    "compile_guard_end": "#endif"
                    if operation.tile_description.math_instruction.opcode_class
                    == OpcodeClass.WmmaTensorOp
                    else "",
                },
            )
        )

    def __exit__(self, exception_type, exception_value, traceback):

        # Write instance definitions in top-level namespace
        for instance_definition in self.instance_definitions:
            self.configuration_file.write(instance_definition)

        # Add wrapper objects within initialize() function
        self.configuration_file.write(
            SubstituteTemplate(
                self.initialize_function_template, {"configuration_name": self.configuration_name}
            )
        )

        for instance_wrapper in self.instance_wrappers:
            self.configuration_file.write(instance_wrapper)

        self.configuration_file.write(self.epilogue_template)
        self.configuration_file.close()


###################################################################################################
###################################################################################################
