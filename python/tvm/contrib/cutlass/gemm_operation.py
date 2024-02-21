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
# pylint: disable=invalid-name, unused-wildcard-import, wildcard-import, pointless-exception-statement
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

        return f"{self.short_math_name()}{inst_shape}{intermediate_type}gemm"

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
        return f"{ShortLayoutTypeNames[self.A.layout]}{ShortLayoutTypeNames[self.B.layout]}"

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
                "alignment": f"{self.A.alignment}",
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
            {"lda_val": lda, "ldb_val": ldb, "ldc_val": ldc},
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

        self.epilogue_residual_block = """
    ${epilogue_functor}<
      ${element_c},
      ${element_accumulator},
      ${element_epilogue},
      ${element_c},
      ${epilogue_vector_length},
      ${activation},
      ${binary_op},
      ${unary_op}
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
    ${align_b}
  >;
"""

    def emit(self, operation, no_beta_scaling=False, batched=False, residual_block_info=False):
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
            "arch": f"cutlass::arch::Sm{operation.arch}",
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

        if residual_block_info:
            values["kernel_name"] = "GemmUniversalWithBroadcast"
            template = substitute_template(
                self.gemm_template, {"epilogue": self.epilogue_residual_block}
            )
            values.update(
                {
                    "unary_op": residual_block_info["unary_op"],
                    "binary_op": residual_block_info["binary_op"],
                    "activation": residual_block_info["activation"],
                }
            )
        elif no_beta_scaling:
            template = substitute_template(
                self.gemm_template, {"epilogue": self.epilogue_no_beta_scaling}
            )
        else:
            template = substitute_template(self.gemm_template, {"epilogue": self.epilogue_default})

        return substitute_template(template, values)


def instantiate_gemm_template(attrs):
    """Return CUTLASS host code for GEMM based on a template and the provided attribute map."""

    argument_template_default = """
  typename ${kernel}::Arguments arguments{
   problem_size,
   {static_cast<ElementInputA*>(ptr_a), ${lda}}, ${batch_stride_A}
   {static_cast<ElementInputB*>(ptr_b), ${ldb}}, ${batch_stride_B}
   {static_cast<ElementOutput*>(${ptr_c}), ${c_stride}}, ${batch_stride_C}
   {static_cast<ElementOutput*>(ptr_out), ${ldc}}, ${batch_stride_D}
   {${alpha_beta}},
   ${split_k_slices_or_batch}
  };
    """

    # See cutlass/gemm/kernel/gemm_with_fused_epilogue.h
    argument_template_residual = """
  typename ${kernel}::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::${gemm_universal_mode},
    problem_size,
    ${split_k_slices_or_batch}, // batch_count
    {${alpha_beta}},
    static_cast<ElementInputA*>(ptr_a),
    static_cast<ElementInputB*>(ptr_b),
    static_cast<ElementOutput*>(ptr_residual),
    static_cast<ElementOutput*>(ptr_out),
    static_cast<ElementOutput*>(ptr_bias),
    nullptr, // ptr_Tensor
    ${batch_stride_A}
    ${batch_stride_B}
    ${batch_stride_C}
    ${batch_stride_D}
    0, // batch_stride_Vector,
    0, // batch_stride_Tensor,
    ${lda},
    ${ldb},
    ${ldc},
    ${ldc},
    0, // ldv, the stride for bias
    0, // ldt
  };
    """

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
  void* ptr_a = (void*)(${lhs_arg}->data);
  void* ptr_b = (void*)(${rhs_arg}->data);
  ${bias_decl}
  ${residual_decl}
  void* ptr_out = (void*)(out0->data);

  ${argument}
  size_t workspace_size = ${kernel}::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  ${kernel} gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  CHECK(status == cutlass::Status::kSuccess);
  status = gemm_op.initialize(arguments, workspace.get());
  CHECK(status == cutlass::Status::kSuccess);

  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

  status = gemm_op(stream);
  CHECK(status == cutlass::Status::kSuccess);
"""
    op_type = attrs["op_type"]
    has_bias = "bias" in op_type
    is_gelu = "gelu" in op_type
    batched = "batch" in attrs
    has_residual_block = "residual" in op_type
    aux_map = {"kernel": "Gemm"}

    if has_bias:
        aux_map.update(
            {
                "bias_decl": "void* ptr_bias = (void*)(${bias_arg}->data);\n",
                "ptr_c": "ptr_bias",
                "c_stride": (
                    "(${bias_arg}->ndim == 1 ||"
                    " ${bias_arg}->shape[${bias_arg}->ndim - 2] == 1) ? 0 : " + attrs["ldc"]
                ),
            }
        )
    else:
        aux_map.update({"bias_decl": "", "ptr_c": "ptr_out", "c_stride": attrs["ldc"]})

    if is_gelu or has_residual_block:
        # GeLU epilogue does not compile with NoBetaScaling, so we explicitly specify the scale.
        aux_map["beta"] = 1
    else:
        aux_map["beta"] = 0

    if has_bias and not is_gelu and not has_residual_block:
        aux_map["alpha_beta"] = "alpha"
    else:
        aux_map["alpha_beta"] = "alpha, beta"

    for key in ["batch_stride_A", "batch_stride_B", "batch_stride_C"]:
        if not batched and not has_residual_block:
            aux_map[key] = ""
        else:
            aux_map[key] = attrs.get(key, "0") + ","

    aux_map["batch_stride_D"] = aux_map["batch_stride_C"]
    if has_bias and batched and not has_residual_block:
        aux_map["batch_stride_C"] = "0,"

    if batched:
        attrs["split_k_slices_or_batch"] = attrs["batch"]
    else:
        attrs["split_k_slices_or_batch"] = 1

    if has_residual_block:
        template = substitute_template(template, {"argument": argument_template_residual})
        aux_map["residual_decl"] = "void* ptr_residual = (void*)(${residual_arg}->data);\n"
        aux_map["gemm_universal_mode"] = "kBatched" if batched else "kGemm"
    else:
        template = substitute_template(template, {"argument": argument_template_default})
        aux_map["residual_decl"] = ""

    template = substitute_template(template, aux_map)

    return substitute_template(template, attrs)


def emit_fp16A_intB_matmul(attrs):
    """Return CUTLASS host code for fp16 A and int4 or int8 B GEMM."""
    if attrs["group_size"] > 0:
        attrs["quant_op"] = "cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY"
    else:
        attrs["quant_op"] = "cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY"
        attrs["group_size"] = "k"

    attrs["template_common"] = substitute_template(
        """
  using namespace fastertransformer;
  constexpr auto QuantOp = ${quant_op};

  int m = ${M};
  int n = ${B_arg}->shape[1] * ${float_per_int};
  int k = ${B_arg}->shape[0];

  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
    """,
        attrs,
    )

    template = """
  ${template_common}
  gemm_fp16_int_bias_act<${weight_dtype}, QuantOp>(static_cast<cutlass::half_t*>(${A_arg}->data),
                static_cast<${weight_dtype}*>(${B_arg}->data),
                static_cast<cutlass::half_t*>(${scales_arg}->data),
                ${bias},
                static_cast<cutlass::half_t*>(out0->data),
                "${activation}",
                m, n, k, ${group_size}, ${bias_stride}, nullptr, 0, stream);
"""

    template_residual = """
  ${template_common}
  gemm_fp16_int_bias_act_residual<${weight_dtype}, QuantOp>(static_cast<cutlass::half_t*>(${A_arg}->data),
                static_cast<${weight_dtype}*>(${B_arg}->data),
                static_cast<cutlass::half_t*>(${scales_arg}->data),
                ${bias},
                static_cast<cutlass::half_t*>(${residual_arg}->data),
                static_cast<cutlass::half_t*>(out0->data), "${activation}", "${binary_op}", "${unary_op}",
                m, n, k, ${group_size}, nullptr, 0, stream);
"""

    if "residual_arg" in attrs:
        if "bias_arg" in attrs:
            bias = "static_cast<cutlass::half_t*>(${bias_arg}->data)"
        else:
            bias = "nullptr"

        template_residual = substitute_template(template_residual, {"bias": bias})
        return substitute_template(template_residual, attrs)

    if "bias_arg" in attrs:
        template = substitute_template(
            template, {"bias": "static_cast<cutlass::half_t*>(${bias_arg}->data)"}
        )
    else:
        template = substitute_template(template, {"bias": "nullptr"})

    return substitute_template(template, attrs)
