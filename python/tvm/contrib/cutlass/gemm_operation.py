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
# pylint: disable=invalid-name, unused-wildcard-import, wildcard-import
"""Generator for CUTLASS GEMM kernels."""
from .library import *


class GemmOperation:
    """Describes various attributes for instantiating GEMM kernels."""

    def __init__(
        self,
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
        self.A = A
        self.B = B
        self.C = C
        self.element_epilogue = element_epilogue
        self.epilogue_functor = epilogue_functor
        self.swizzling_functor = swizzling_functor

    def accumulator_type(self):
        return self.tile_description.math_instruction.element_accumulator

    def short_math_name(self):
        return ShortDataTypeNames[self.accumulator_type()]

    def core_name(self):
        """The basic operation kind is prefixed with a letter indicating the accumulation type."""
        inst_shape = ""
        intermediate_type = ""

        if (
            self.tile_description.math_instruction.opcode_class == OpcodeClass.TensorOp
            or self.tile_description.math_instruction.opcode_class == OpcodeClass.WmmaTensorOp
        ):
            inst_shape = "%d%d%d" % tuple(self.tile_description.math_instruction.instruction_shape)
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
            "gemm",
        )

    def extended_name(self):
        """Append data types if they differ from compute type."""
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

        extended_name = substitute_template(
            extended_name,
            {
                "element_a": DataTypeNames[self.A.element],
                "element_c": DataTypeNames[self.C.element],
                "core_name": self.core_name(),
            },
        )

        return extended_name

    def layout_name(self):
        return "%s%s" % (ShortLayoutTypeNames[self.A.layout], ShortLayoutTypeNames[self.B.layout])

    def procedural_name(self):
        """The full procedural name indicates architecture, extended name, tile size,
        and layout.
        """
        threadblock = self.tile_description.procedural_name()
        opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]

        return substitute_template(
            "cutlass_${opcode_class}_${extended_name}_${threadblock}_${layout}_align${alignment}",
            {
                "opcode_class": opcode_class_name,
                "extended_name": self.extended_name(),
                "threadblock": threadblock,
                "layout": self.layout_name(),
                "alignment": "%d" % self.A.alignment,
            },
        )

    def leading_dim(self):
        """lda, ldb, ldc, according to the leading dimension."""
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

        return substitute_template(
            "int lda = ${lda_val};\n\tint ldb = ${ldb_val};\n\tint ldc = ${ldc_val};\n",
            {
                "lda_val": lda,
                "ldb_val": ldb,
                "ldc_val": ldc,
            },
        )


class EmitGemmInstance:
    """Responsible for emitting a CUTLASS template definition."""

    def __init__(self):
        self.epilogue_default = """
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >"""
        self.epilogue_no_beta_scaling = """
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue},
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >"""
        self.gemm_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::${kernel_name}<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue},
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    ${split_k_serial}
    ${math_operation}
  >;
"""

    def emit(self, operation, no_beta_scaling=False, batched=False):
        """Instantiate a GEMM kernel from given `operation`."""
        warp_shape = [
            operation.tile_description.threadblock_shape[idx]
            // operation.tile_description.warp_count[idx]
            for idx in range(3)
        ]
        epilogue_vector_length = (
            min(operation.C.alignment * DataTypeSize[operation.C.element], 128)
            // DataTypeSize[operation.C.element]
        )
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
            "math_operation": MathOperationTag[
                operation.tile_description.math_instruction.math_operation
            ],
        }

        values["kernel_name"] = "GemmBatched" if batched else "Gemm"
        values["split_k_serial"] = "" if batched else "false,"

        gemm_template = substitute_template(
            self.gemm_template,
            {
                "epilogue": self.epilogue_no_beta_scaling
                if no_beta_scaling
                else self.epilogue_default
            },
        )
        return substitute_template(gemm_template, values)


def instantiate_gemm_template(attrs, func_args):
    """Return CUTLASS host code for GEMM based on a template and the provided attribute map."""

    template = """
  using ElementInputA = ${ElementInputA};
  using ElementInputB = ${ElementInputB};
  using ElementOutput = ${ElementOutput};
  using ElementComputeEpilogue = ${ElementOutput};

  ${cutlass_op_def}

  using ${kernel} = Operation_${cutlass_op_name};
  int M = ${M};
  int N = ${N};
  int K = ${K};
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(${beta});
  void* ptr_a = (void*)(${arg0}->data);
  void* ptr_b = (void*)(${arg1}->data);
  ${bias_decl}
  void* ptr_out = (void*)(out0->data);

  typename ${kernel}::Arguments arguments{
   problem_size,
   {static_cast<ElementInputA*>(ptr_a), ${lda}}, ${batch_stride_A}
   {static_cast<ElementInputB*>(ptr_b), ${ldb}}, ${batch_stride_B}
   {static_cast<ElementOutput*>(${ptr_c}), ${c_stride}}, ${batch_stride_C}
   {static_cast<ElementOutput*>(ptr_out), ${ldc}}, ${batch_stride_C}
   {${alpha_beta}},
   ${split_k_slices_or_batch}
  };
  size_t workspace_size = ${kernel}::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  ${kernel} gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  CHECK(status == cutlass::Status::kSuccess);
  status = gemm_op.initialize(arguments, workspace.get());
  CHECK(status == cutlass::Status::kSuccess);
  status = gemm_op();
  CHECK(status == cutlass::Status::kSuccess);
"""
    has_bias = "bias" in attrs["op_type"]
    is_gelu = "gelu" in attrs["op_type"]
    batched = "batch_matmul" in attrs["op_type"]

    aux_map = {"kernel": "Gemm"}

    if has_bias:
        aux_map.update(
            {
                "bias_decl": "void* ptr_c_bias = (void*)(${arg2}->data);\n",
                "ptr_c": "ptr_c_bias",
                "c_stride": "0",
            }
        )
    else:
        aux_map.update({"bias_decl": "", "ptr_c": "ptr_out", "c_stride": attrs["ldc"]})

    if is_gelu:
        # GeLU epilogue does not compile with NoBetaScaling, so we explicitly specify the scale.
        aux_map["beta"] = "1"
    else:
        aux_map["beta"] = "0"

    if has_bias and not is_gelu:
        aux_map["alpha_beta"] = "alpha"
    else:
        aux_map["alpha_beta"] = "alpha, beta"

    for key in ["batch_stride_A", "batch_stride_B", "batch_stride_C"]:
        if not batched:
            aux_map[key] = ""
        else:
            aux_map[key] = attrs[key] + ","

    if batched:
        attrs["split_k_slices_or_batch"] = attrs["batch"]
    else:
        attrs["split_k_slices_or_batch"] = "1"

    template = substitute_template(template, aux_map)

    for i, arg in enumerate(func_args):
        attrs["arg{}".format(i)] = arg

    return substitute_template(template, attrs)
