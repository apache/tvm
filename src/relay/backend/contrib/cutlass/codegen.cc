/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/backend/contrib/cutlass/codegen.cc
 * \brief The 'custom' compilation pass for CUTLASS (invoked by the RelayToTIRTargetHook pass).
 */

#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <numeric>
#include <sstream>

#include "../../../transforms/compiler_function_utils.h"
#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cutlass {

namespace {

/*! \brief Return the "cutlass" Target instance to use to guide compilation. */
Target GetCutlassTarget() {
  Target target = Target::Current(/*allow_not_defined=*/true);
  if (!target.defined() || target->kind->name != "cutlass") {
    // Use the default CUTLASS compilation options if no specific "cutlass" target was given
    // in the overall targets list. In that case target_hooks.cc will invoke the custom pass
    // without pushing any target instance onto the implicit target stack.
    target = Target("cutlass");
  }
  return target;
}

using Str2StrMap = std::unordered_map<std::string, std::string>;

static Str2StrMap dtype_map = {{"float16", "cutlass::half_t"},
                               {"float32", "float"},
                               {"int8", "int8_t"},
                               {"uint8", "uint8_t"},
                               {"int32", "int32_t"}};

constexpr const char* kAnyDim = "Any";

std::string GetDimAsStr(ObjectRef dim) {
  if (auto d = dim.as<IntImmNode>()) {
    return std::to_string(d->value);
  }
  return kAnyDim;
}

inline void CutlassPrint(std::ostringstream& os, const std::string& stmt, int indent = 2) {
  for (int i = 0; i < indent; ++i) {
    os << " ";
  }
  os << stmt;
}

Str2StrMap ArgsCommon(const Map<String, ObjectRef>& attrs) {
  Str2StrMap args;
  auto arg0_dtype = std::string(attrs["arg0_dtype"].as<StringObj>()->data);
  auto arg1_dtype = std::string(attrs["arg1_dtype"].as<StringObj>()->data);
  auto ret_dtype = std::string(attrs["ret_dtype"].as<StringObj>()->data);
  args["ElementInputA"] = dtype_map.at(arg0_dtype);
  args["ElementInputB"] = dtype_map.at(arg1_dtype);
  args["ElementOutput"] = dtype_map.at(ret_dtype);
  args["op_def"] = std::string(attrs["cutlass_op_def"].as<StringObj>()->data);
  args["op_name"] = std::string(attrs["cutlass_op_name"].as<StringObj>()->data);
  args["op_type"] = std::string(attrs["op_type"].as<StringObj>()->data);
  return args;
}

Str2StrMap GemmArgsCommon(const Map<String, ObjectRef>& attrs) {
  Str2StrMap args = ArgsCommon(attrs);
  args["lda"] = std::string(attrs["lda"].as<StringObj>()->data);
  args["ldb"] = std::string(attrs["ldb"].as<StringObj>()->data);
  args["ldc"] = std::string(attrs["ldc"].as<StringObj>()->data);
  return args;
}

Str2StrMap DenseArgs(const Map<String, ObjectRef>& attrs) {
  Str2StrMap args = GemmArgsCommon(attrs);
  auto arg0_shape = attrs["arg0_shape"].as<ArrayNode>();
  auto arg1_shape = attrs["arg1_shape"].as<ArrayNode>();
  args["M"] = GetDimAsStr(arg0_shape->at(0));
  args["K"] = GetDimAsStr(arg0_shape->at(1));
  args["N"] = GetDimAsStr(arg1_shape->at(0));
  return args;
}

Str2StrMap BatchMatmulArgs(const Map<String, ObjectRef>& attrs) {
  Str2StrMap args = GemmArgsCommon(attrs);
  args["batch"] = GetDimAsStr(attrs["batch"]);
  args["batch_stride_A"] = GetDimAsStr(attrs["batch_stride_A"]);
  args["batch_stride_B"] = GetDimAsStr(attrs["batch_stride_B"]);
  args["batch_stride_C"] = GetDimAsStr(attrs["batch_stride_C"]);
  auto arg0_shape = attrs["arg0_shape"].as<ArrayNode>();
  auto arg1_shape = attrs["arg1_shape"].as<ArrayNode>();
  args["M"] = GetDimAsStr(arg0_shape->at(1));
  args["K"] = GetDimAsStr(arg0_shape->at(2));
  args["N"] = GetDimAsStr(arg1_shape->at(1));
  return args;
}

void AppendPrologue(std::ostringstream& gemm_decl, const Str2StrMap& attrs,
                    const std::vector<std::string>& func_args, const std::string& kernel,
                    bool has_bias, bool is_gelu, int m_axis_idx, int n_axis_idx, int k_axis_idx) {
  CutlassPrint(gemm_decl, "using ElementInputA = " + attrs.at("ElementInputA") + ";\n");
  CutlassPrint(gemm_decl, "using ElementInputB = " + attrs.at("ElementInputB") + ";\n");
  CutlassPrint(gemm_decl, "using ElementOutput = " + attrs.at("ElementOutput") + ";\n");
  CutlassPrint(gemm_decl, "using ElementComputeEpilogue = " + attrs.at("ElementOutput") + ";\n");
  CutlassPrint(gemm_decl, attrs.at("op_def"));
  CutlassPrint(gemm_decl, "using " + kernel + " = Operation_" + attrs.at("op_name") + ";\n");

  auto get_dim = [&attrs, &func_args](const std::string& axis, int arg_idx, int axis_idx) {
    if (attrs.at(axis) == kAnyDim) {
      return func_args[arg_idx] + "->shape[" + std::to_string(axis_idx) + "]";
    } else {
      return attrs.at(axis);
    }
  };
  CutlassPrint(gemm_decl, "int M = " + get_dim("M", 0, m_axis_idx) + ";\n");
  CutlassPrint(gemm_decl, "int N = " + get_dim("N", 1, n_axis_idx) + ";\n");
  CutlassPrint(gemm_decl, "int K = " + get_dim("K", 0, k_axis_idx) + ";\n");
  CutlassPrint(gemm_decl, "cutlass::gemm::GemmCoord problem_size(M, N, K);\n");
  CutlassPrint(gemm_decl, "ElementComputeEpilogue alpha = ElementComputeEpilogue(1);\n");
  if (is_gelu) {
    // GeLU epilogue does not compile with NoBetaScaling, so we explicitly specify the scale.
    CutlassPrint(gemm_decl, "ElementComputeEpilogue beta = ElementComputeEpilogue(1);\n");
  } else {
    CutlassPrint(gemm_decl, "ElementComputeEpilogue beta = ElementComputeEpilogue(0);\n");
  }

  ICHECK(func_args.size() >= 2);
  CutlassPrint(gemm_decl, "void* ptr_a = (void*)(" + func_args[0] + "->data);\n");
  CutlassPrint(gemm_decl, "void* ptr_b = (void*)(" + func_args[1] + "->data);\n");
  if (has_bias) {
    ICHECK(func_args.size() >= 3);
    CutlassPrint(gemm_decl, "void* ptr_c_bias = (void*)(" + func_args[2] + "->data);\n");
  }

  CutlassPrint(gemm_decl, "void* ptr_out = (void*)(out0->data);\n");

  CutlassPrint(gemm_decl, "typename " + kernel + "::Arguments arguments{\n");
  CutlassPrint(gemm_decl, " problem_size,\n");
}

void AppendGemmExecute(std::ostringstream& gemm_decl, const std::string& kernel) {
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  CutlassPrint(gemm_decl,
               "size_t workspace_size = " + kernel + "::get_workspace_size(arguments);\n");
  // Allocate workspace memory
  CutlassPrint(gemm_decl,
               "cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);\n");
  // Instantiate CUTLASS kernel depending on template
  CutlassPrint(gemm_decl, kernel + " gemm_op;\n");

  // Check the problem size is supported or not
  CutlassPrint(gemm_decl, "cutlass::Status status = gemm_op.can_implement(arguments);\n");
  CutlassPrint(gemm_decl, "CHECK(status == cutlass::Status::kSuccess);\n");
  // Initialize CUTLASS kernel with arguments and workspace pointer
  CutlassPrint(gemm_decl, "status = gemm_op.initialize(arguments, workspace.get());\n");
  CutlassPrint(gemm_decl, "CHECK(status == cutlass::Status::kSuccess);\n");
  // Launch initialized CUTLASS kernel
  CutlassPrint(gemm_decl, "status = gemm_op();\n");
  CutlassPrint(gemm_decl, "CHECK(status == cutlass::Status::kSuccess);\n");
}

std::string DenseOp(std::string id, const Str2StrMap& attrs,
                    const std::vector<std::string>& func_args) {
  bool has_bias = attrs.at("op_type").find("bias") != std::string::npos;
  bool is_gelu =
      attrs.at("op_type").find("cutlass.dense_bias_gelu") != std::string::npos;  // fp32 or fp16
  std::ostringstream gemm_decl;
  AppendPrologue(gemm_decl, attrs, func_args, "Gemm", has_bias, is_gelu, 0, 0, 1);

  CutlassPrint(gemm_decl, " {static_cast<ElementInputA*>(ptr_a), " + attrs.at("lda") + "},\n");
  CutlassPrint(gemm_decl, " {static_cast<ElementInputB*>(ptr_b), " + attrs.at("ldb") + "},\n");
  if (has_bias) {
    CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_c_bias), 0},\n");
  } else {
    CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_out), " + attrs.at("ldc") + "},\n");
  }
  CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_out), " + attrs.at("ldc") + "},\n");
  if (has_bias && !is_gelu) {
    CutlassPrint(gemm_decl, " {alpha},\n");
  } else {
    // For GeLU, we explicitly specify the scale.
    CutlassPrint(gemm_decl, " {alpha, beta},\n");
  }
  CutlassPrint(gemm_decl, " 1};\n");  // split_k_slices

  AppendGemmExecute(gemm_decl, "Gemm");
  return gemm_decl.str();
}

std::string BatchMatmulOp(std::string id, const Str2StrMap& attrs,
                          const std::vector<std::string>& func_args) {
  std::ostringstream gemm_decl;
  AppendPrologue(gemm_decl, attrs, func_args, "BatchedGemm", false, false, 1, 1, 2);

  auto get_batch_stride = [&attrs, &func_args](const std::string& name, int arg0_idx, int arg1_idx,
                                               int arg0_axis_idx, int arg1_axis_idx) {
    if (attrs.at(name) == kAnyDim) {
      return func_args[arg0_idx] + "->shape[" + std::to_string(arg0_axis_idx) + "] * " +
             func_args[arg1_idx] + "->shape[" + std::to_string(arg1_axis_idx) + "]";
    } else {
      return attrs.at(name);
    }
  };

  CutlassPrint(gemm_decl, " {static_cast<ElementInputA*>(ptr_a), " + attrs.at("lda") + "},\n");
  CutlassPrint(gemm_decl, get_batch_stride("batch_stride_A", 0, 0, 1, 2) + ",\n");
  CutlassPrint(gemm_decl, " {static_cast<ElementInputB*>(ptr_b), " + attrs.at("ldb") + "},\n");
  CutlassPrint(gemm_decl, get_batch_stride("batch_stride_B", 1, 1, 1, 2) + ",\n");
  CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_out), " + attrs.at("ldc") + "},\n");
  CutlassPrint(gemm_decl, get_batch_stride("batch_stride_C", 0, 1, 1, 1) + ",\n");
  CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_out), " + attrs.at("ldc") + "},\n");
  CutlassPrint(gemm_decl, get_batch_stride("batch_stride_C", 0, 1, 1, 1) + ",\n");
  CutlassPrint(gemm_decl, " {alpha, beta},\n");

  if (attrs.at("batch") == kAnyDim) {
    CutlassPrint(gemm_decl, func_args[0] + "->shape[0]" + "};\n");
  } else {
    CutlassPrint(gemm_decl, attrs.at("batch") + "};\n");
  }

  AppendGemmExecute(gemm_decl, "BatchedGemm");
  return gemm_decl.str();
}

Str2StrMap Conv2dArgs(const Map<String, ObjectRef>& attrs, bool is_dgrad = false,
                      bool is_wgrad = false) {
  Str2StrMap args = ArgsCommon(attrs);
  auto arg0_shape = attrs["arg0_shape"].as<ArrayNode>();
  auto arg1_shape = attrs["arg1_shape"].as<ArrayNode>();
  auto ret_shape = attrs["ret_shape"].as<ArrayNode>();
  auto activation_shape = arg0_shape;
  auto weight_shape = arg1_shape;
  auto output_shape = ret_shape;

  if (is_dgrad) {
    activation_shape = ret_shape;
    output_shape = arg0_shape;
  } else if (is_wgrad) {
    activation_shape = arg1_shape;
    weight_shape = ret_shape;
    output_shape = arg0_shape;
  }

  args["N"] = GetDimAsStr(activation_shape->at(0));
  args["H"] = GetDimAsStr(activation_shape->at(1));
  args["W"] = GetDimAsStr(activation_shape->at(2));
  args["C"] = GetDimAsStr(activation_shape->at(3));
  args["P"] = GetDimAsStr(output_shape->at(1));
  args["Q"] = GetDimAsStr(output_shape->at(2));
  args["K"] = GetDimAsStr(output_shape->at(3));
  args["R"] = GetDimAsStr(weight_shape->at(1));
  args["S"] = GetDimAsStr(weight_shape->at(2));
  args["pad_h"] = GetDimAsStr(attrs["padding"].as<ArrayNode>()->at(0));
  args["pad_w"] = GetDimAsStr(attrs["padding"].as<ArrayNode>()->at(1));
  args["stride_h"] = GetDimAsStr(attrs["strides"].as<ArrayNode>()->at(0));
  args["stride_w"] = GetDimAsStr(attrs["strides"].as<ArrayNode>()->at(1));
  args["dilation_h"] = GetDimAsStr(attrs["dilation"].as<ArrayNode>()->at(0));
  args["dilation_w"] = GetDimAsStr(attrs["dilation"].as<ArrayNode>()->at(1));

  return args;
}

std::string Conv2dOp(std::string id, const Str2StrMap& attrs,
                     const std::vector<std::string>& func_args, bool has_residual_block = false) {
  auto op_type = attrs.at("op_type");
  bool has_bias = op_type.find("bias") != std::string::npos;
  bool no_bias_scaling = op_type != "cutlass.conv2d_bias_sigmoid" &&
                         op_type != "cutlass.conv2d_bias_silu" &&
                         op_type != "cutlass.conv2d_bias_hardswish";

  const std::string op_name = attrs.at("op_name");
  std::ostringstream conv2d_decl;
  CutlassPrint(conv2d_decl, attrs.at("op_def"));
  CutlassPrint(conv2d_decl, "using Operation_" + op_name +
                                " = cutlass::conv::device::ImplicitGemmConvolution<" + op_name +
                                ">;\n");
  CutlassPrint(conv2d_decl, "using Conv2d = Operation_" + op_name + ";\n");
  CutlassPrint(conv2d_decl, "using ElementInputA = Conv2d::ElementA;\n");
  CutlassPrint(conv2d_decl, "using ElementInputB = Conv2d::ElementB;\n");
  CutlassPrint(conv2d_decl, "using ElementComputeEpilogue = Conv2d::ElementAccumulator;\n");

  auto get_dim = [&attrs](const std::string& axis, const std::string& var_name, int axis_idx) {
    if (attrs.at(axis) == kAnyDim) {
      return var_name + "->shape[" + std::to_string(axis_idx) + "]";
    } else {
      return attrs.at(axis);
    }
  };

  CutlassPrint(conv2d_decl, "int N = " + get_dim("N", func_args[0], 0) + ";\n");
  CutlassPrint(conv2d_decl, "int H = " + get_dim("H", func_args[0], 1) + ";\n");
  CutlassPrint(conv2d_decl, "int W = " + get_dim("W", func_args[0], 2) + ";\n");
  CutlassPrint(conv2d_decl, "int C = " + attrs.at("C") + ";\n");
  CutlassPrint(conv2d_decl, "int K = " + attrs.at("K") + ";\n");
  CutlassPrint(conv2d_decl, "int R = " + attrs.at("R") + ";\n");
  CutlassPrint(conv2d_decl, "int S = " + attrs.at("S") + ";\n");
  CutlassPrint(conv2d_decl, "int P = " + get_dim("P", "out0", 1) + ";\n");
  CutlassPrint(conv2d_decl, "int Q = " + get_dim("Q", "out0", 2) + ";\n");
  CutlassPrint(conv2d_decl, "int pad_h = " + attrs.at("pad_h") + ";\n");
  CutlassPrint(conv2d_decl, "int pad_w = " + attrs.at("pad_w") + ";\n");
  CutlassPrint(conv2d_decl, "int stride_h = " + attrs.at("stride_h") + ";\n");
  CutlassPrint(conv2d_decl, "int stride_w = " + attrs.at("stride_w") + ";\n");
  CutlassPrint(conv2d_decl, "int dilation_h = " + attrs.at("dilation_h") + ";\n");
  CutlassPrint(conv2d_decl, "int dilation_w = " + attrs.at("dilation_w") + ";\n");

  const bool use_split_k = op_name.find("splitk") != std::string::npos;

  if (use_split_k) {
    std::string split_k_slices = op_name.substr(op_name.find_last_not_of("0123456789") + 1);
    CutlassPrint(conv2d_decl, "int split_k_slices = " + split_k_slices + ";\n");
  } else {
    CutlassPrint(conv2d_decl, "int split_k_slices = 1;\n");
  }

  CutlassPrint(
      conv2d_decl,
      "cutlass::conv::Conv2dProblemSize problem_size(N, H, W, C, K, R, S, P, Q, pad_h, pad_w, "
      "stride_h, stride_w, dilation_h, dilation_w, cutlass::conv::Mode::kCrossCorrelation, "
      "split_k_slices);\n");

  const std::string split_k_mode = use_split_k ? "kParallel" : "kSerial";
  CutlassPrint(conv2d_decl,
               "const cutlass::conv::SplitKMode split_k_mode = cutlass::conv::SplitKMode::" +
                   split_k_mode + ";\n");

  bool is_wgrad = op_type.find("backward_weight") != std::string::npos;
  bool is_dgrad = op_type.find("conv2d_transpose") != std::string::npos;

  ICHECK(func_args.size() >= 2);
  CutlassPrint(conv2d_decl, "void* ptr_a = (void*)(" + func_args[0] + "->data);\n");
  CutlassPrint(conv2d_decl, "void* ptr_b = (void*)(" + func_args[1] + "->data);\n");

  if (has_residual_block) {
    ICHECK(func_args.size() >= 4);
    CutlassPrint(conv2d_decl, "void* ptr_bias = (void*)(" + func_args[2] + "->data);\n");
    CutlassPrint(conv2d_decl, "void* ptr_residual = (void*)(" + func_args[3] + "->data);\n");
  } else if (has_bias) {
    ICHECK(func_args.size() >= 3);
    CutlassPrint(conv2d_decl, "void* ptr_c_bias = (void*)(" + func_args[2] + "->data);\n");
  }

  CutlassPrint(conv2d_decl, "void* ptr_out = (void*)(out0->data);\n");
  CutlassPrint(conv2d_decl, "ElementComputeEpilogue alpha = ElementComputeEpilogue(1);\n");
  if ((!has_bias || no_bias_scaling) && !has_residual_block) {
    CutlassPrint(conv2d_decl, "ElementComputeEpilogue beta = ElementComputeEpilogue(0);\n");
  } else {
    CutlassPrint(conv2d_decl, "ElementComputeEpilogue beta = ElementComputeEpilogue(1);\n");
  }
  CutlassPrint(conv2d_decl, "using cutlass::layout::TensorNHWC;\n");
  CutlassPrint(conv2d_decl,
               "auto activation_shape = TensorNHWC::packed(cutlass::make_Coord(N, H, W, C));\n");
  CutlassPrint(conv2d_decl,
               "auto weight_shape = TensorNHWC::packed(cutlass::make_Coord(K, R, S, C));\n");
  CutlassPrint(conv2d_decl,
               "auto output_oshape = TensorNHWC::packed(cutlass::make_Coord(N, P, Q, K));\n");

  if (is_wgrad) {
    CutlassPrint(conv2d_decl, "TensorNHWC layout_A(output_oshape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_B(activation_shape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_C(weight_shape);\n\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_D(weight_shape);\n\n");
  } else if (is_dgrad) {
    CutlassPrint(conv2d_decl, "TensorNHWC layout_A(output_oshape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_B(weight_shape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_C(activation_shape);\n\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_D(activation_shape);\n\n");
  } else {
    CutlassPrint(conv2d_decl, "TensorNHWC layout_A(activation_shape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_B(weight_shape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_C(output_oshape);\n\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_D(output_oshape);\n\n");
  }

  if (use_split_k) {
    CutlassPrint(conv2d_decl, "using ElementOutput = EpilogueOutputOp::ElementOutput;\n");
  } else {
    CutlassPrint(conv2d_decl, "using ElementOutput = Conv2d::ElementC;\n");
  }

  std::string tensor_c_init = "{static_cast<ElementOutput*>(ptr_out), layout_C}";
  if (has_residual_block) {
    tensor_c_init = "{static_cast<ElementOutput*>(ptr_residual), layout_C}";
  } else if (has_bias) {
    tensor_c_init =
        "{static_cast<ElementOutput*>(ptr_c_bias), cutlass::layout::TensorNHWC::Stride(0)}";
  }

  CutlassPrint(conv2d_decl,
               "cutlass::TensorRef<ElementOutput, TensorNHWC> tensor_c" + tensor_c_init + ";\n");
  CutlassPrint(conv2d_decl,
               "cutlass::TensorRef<ElementOutput, TensorNHWC> "
               "tensor_d{static_cast<ElementOutput*>(ptr_out),layout_D};\n");

  CutlassPrint(conv2d_decl, "typename Conv2d::Arguments arguments{\n");
  CutlassPrint(conv2d_decl, " problem_size,\n");
  CutlassPrint(conv2d_decl, " {static_cast<ElementInputA*>(ptr_a), layout_A},\n");
  CutlassPrint(conv2d_decl, " {static_cast<ElementInputB*>(ptr_b), layout_B},\n");

  if (use_split_k) {
    CutlassPrint(conv2d_decl, "{nullptr, TensorNHWC()},\n");
    CutlassPrint(conv2d_decl, "{nullptr, TensorNHWC()},\n");
  } else {
    CutlassPrint(conv2d_decl, " tensor_c,\n");
    CutlassPrint(conv2d_decl, " tensor_d,\n");
  }

  if (has_residual_block) {
    ICHECK(use_split_k == false) << "Split-k not supported for residual block fusion";
    CutlassPrint(conv2d_decl, "{alpha, beta},\n");
    CutlassPrint(conv2d_decl, "cutlass::conv::SplitKMode::kSerial,\n");  // split_k_slices
    CutlassPrint(conv2d_decl, "static_cast<ElementOutput*>(ptr_bias),\n");
    CutlassPrint(conv2d_decl, "nullptr, 0, K};\n");
  } else if (has_bias && no_bias_scaling) {
    CutlassPrint(conv2d_decl, " {alpha},\n");
    CutlassPrint(conv2d_decl, "split_k_mode\n};\n");
  } else {
    CutlassPrint(conv2d_decl, "{alpha, beta},\n");
    CutlassPrint(conv2d_decl, "split_k_mode\n};\n");
  }

  CutlassPrint(conv2d_decl, "Conv2d conv2d_op;\n");

  CutlassPrint(conv2d_decl, "size_t workspace_size = conv2d_op.get_workspace_size(arguments);\n");
  // Allocate workspace memory
  CutlassPrint(conv2d_decl,
               "cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);\n");
  // Check the problem size is supported or not
  CutlassPrint(conv2d_decl, "cutlass::Status status = conv2d_op.can_implement(arguments);\n");
  CutlassPrint(conv2d_decl, "CHECK(status == cutlass::Status::kSuccess);\n\n");

  if (use_split_k) {
    CutlassPrint(conv2d_decl,
                 "arguments.ref_D.reset(reinterpret_cast<ElementComputeEpilogue*>(workspace.get()),"
                 " layout_D);\n\n");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CutlassPrint(conv2d_decl, "status = conv2d_op.initialize(arguments, workspace.get());\n");
  CutlassPrint(conv2d_decl, "CHECK(status == cutlass::Status::kSuccess);\n\n");

  if (use_split_k) {
    CutlassPrint(
        conv2d_decl,
        "arguments.output_op = {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}; \n");
    CutlassPrint(conv2d_decl, "status = conv2d_op.update(arguments, workspace.get()); \n");
    CutlassPrint(conv2d_decl, "CHECK(status == cutlass::Status::kSuccess);\n\n");
  }

  // Launch initialized CUTLASS kernel
  CutlassPrint(conv2d_decl, "status = conv2d_op();\n");
  CutlassPrint(conv2d_decl, "CHECK(status == cutlass::Status::kSuccess);\n\n");

  if (use_split_k) {
    CutlassPrint(conv2d_decl, "ReductionDevice reduction_op;\n");
    CutlassPrint(conv2d_decl,
                 "const static cutlass::conv::Operator kConvolutionalOperator = "
                 "Conv2d::kConvolutionalOperator;\n");
    CutlassPrint(conv2d_decl, "typename ReductionDevice::Arguments reduction_args(\n");
    CutlassPrint(conv2d_decl,
                 "cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, "
                 "problem_size).mn(),\n");
    CutlassPrint(conv2d_decl, "problem_size.split_k_slices,\n");
    CutlassPrint(conv2d_decl,
                 "cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, "
                 "problem_size),\n");
    CutlassPrint(conv2d_decl, "{\n");
    CutlassPrint(conv2d_decl,
                 " reinterpret_cast<Conv2d::ElementAccumulator*> (workspace.get()),\n");
    CutlassPrint(conv2d_decl,
                 "ReductionStrideIndex(tensor_c.stride()[Conv2d::ImplicitGemmKernel::"
                 "kTensorCStrideIdx])\n");
    CutlassPrint(conv2d_decl, "},\n");
    CutlassPrint(conv2d_decl, "{\n");
    CutlassPrint(conv2d_decl, "tensor_d.data(),\n");
    CutlassPrint(conv2d_decl,
                 "ReductionStrideIndex(tensor_d.stride()[Conv2d::ImplicitGemmKernel::"
                 "kTensorCStrideIdx])\n");
    CutlassPrint(conv2d_decl, "},\n");
    CutlassPrint(conv2d_decl, "{\n");
    CutlassPrint(conv2d_decl, "tensor_c.data(),\n");
    CutlassPrint(conv2d_decl,
                 "ReductionStrideIndex(tensor_c.stride()[Conv2d::ImplicitGemmKernel::"
                 "kTensorCStrideIdx])\n");
    CutlassPrint(conv2d_decl, "},\n");
    CutlassPrint(conv2d_decl, "   {alpha, beta}\n");
    CutlassPrint(conv2d_decl, ");\n\n");
    CutlassPrint(conv2d_decl, "status = reduction_op.initialize(reduction_args, nullptr);\n");
    CutlassPrint(conv2d_decl, "status = reduction_op();\n");
  }

  return conv2d_decl.str();
}

class CodegenCutlass : public backend::MemoizedExprTranslator<std::vector<Output>>,
                       public CodegenCBase {
 public:
  CodegenCutlass(const std::string& id, const Map<String, ObjectRef>& attrs) {
    this->ext_func_id_ = id;
    this->attrs_ = attrs;
  }

  std::vector<Output> VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "Cutlass codegen doesn't support: " << op->GetTypeKey();
    return {};
  }

  std::vector<Output> VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    const auto* func = call->op.as<FunctionNode>();
    ICHECK(func) << "Only composite function is supported for CUTLASS.";
    GenerateBodyOutput ret = GenerateCompositeFunctionCall(func, call);
    ext_func_body_.push_back(ret.decl);
    return ret.outputs;
  }

  std::string JIT(const std::vector<Output>& out) {
    code_stream_ << "void " << ext_func_id_ << "_(";

    for (const auto& arg : ext_func_args_) {
      code_stream_ << "DLTensor* " << arg->name_hint() << ", ";
    }
    for (size_t i = 0; i < out.size() - 1; ++i) {
      code_stream_ << "DLTensor* out" << i << ", ";
    }
    code_stream_ << "DLTensor* out" << out.size() - 1 << ") {\n";
    this->EnterScope();

    // Function body
    for (auto decl : buf_decl_) {
      this->PrintIndents();
      code_stream_ << decl << "\n";
    }
    code_stream_ << "\n";
    for (auto stmt : ext_func_body_) {
      this->PrintIndents();
      code_stream_ << stmt << "\n";
    }

    this->ExitScope();
    code_stream_ << "}\n";

    this->GenerateBackendCFunc(ext_func_id_, ext_func_args_, /*const_arr_name=*/"", out, true);
    return code_stream_.str();
  }

 private:
  std::vector<std::string> GetArgumentNames(const CallNode* call) {
    std::vector<std::string> arg_names;
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (const auto& out : res) {
        arg_names.push_back(out.name);
      }
    }
    return arg_names;
  }

  bool IsConv2dResidualBlock(const std::string& func_name) {
    return func_name.find("conv2d") != std::string::npos &&
           func_name.find("residual") != std::string::npos;
  }

  // Is node `x` an ancestor of `y`?
  bool IsAncestor(const CallNode* x, const CallNode* y) {
    if (x == y) return true;
    for (auto arg : y->args) {
      const CallNode* arg_ptr = arg.as<CallNode>();
      if (arg_ptr && IsAncestor(x, arg_ptr)) return true;
    }
    return false;
  }

  GenerateBodyOutput GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                   const CallNode* caller) {
    using backend::GetRootCall;

    const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
    ICHECK(pattern_name.defined()) << "Only functions with composite attribute are supported.";

    if (pattern_name == "cutlass.dense") {
      const auto* dense_call =
          GetRootCall(callee->body.as<CallNode>(), 0, std::vector<std::string>{"nn.dense"});
      return GenerateBody(dense_call, "cutlass_dense", GetArgumentNames(caller),
                          DenseArgs(std::ref(attrs_)));
    } else if (pattern_name == "cutlass.dense_bias") {
      const CallNode* current_call = callee->body.as<CallNode>();
      std::string add_or_bias_add = current_call->op.as<OpNode>()->name;
      const auto* dense_call =
          GetRootCall(callee->body.as<CallNode>(), 1, {"nn.dense", add_or_bias_add});
      return GenerateBody(dense_call, "cutlass_dense_bias", GetArgumentNames(caller),
                          DenseArgs(std::ref(attrs_)));
    } else if (pattern_name == "cutlass.dense_bias_relu") {
      const CallNode* current_call = callee->body.as<CallNode>();
      std::string add_or_bias_add = current_call->args[0].as<CallNode>()->op.as<OpNode>()->name;
      const auto* dense_call =
          GetRootCall(callee->body.as<CallNode>(), 2, {"nn.dense", add_or_bias_add, "nn.relu"});
      return GenerateBody(dense_call, "cutlass_dense_bias_relu", GetArgumentNames(caller),
                          DenseArgs(std::ref(attrs_)));
    } else if (pattern_name == "cutlass.dense_bias_gelu_fp16") {
      const CallNode* current_call = callee->body.as<CallNode>();
      std::string add_or_bias_add = current_call->args[1].as<CallNode>()->op.as<OpNode>()->name;
      const auto* dense_call = GetRootCall(callee->body.as<CallNode>(), 8,
                                           {"nn.dense", add_or_bias_add, "multiply", "cast", "erf",
                                            "cast", "multiply", "add", "multiply"});
      return GenerateBody(dense_call, "cutlass_dense_bias_gelu", GetArgumentNames(caller),
                          DenseArgs(std::ref(attrs_)));
    } else if (pattern_name == "cutlass.dense_bias_gelu_fp32") {
      const CallNode* current_call = callee->body.as<CallNode>();
      std::string add_or_bias_add = current_call->args[1].as<CallNode>()->op.as<OpNode>()->name;
      const auto* dense_call = GetRootCall(
          callee->body.as<CallNode>(), 6,
          {"nn.dense", add_or_bias_add, "multiply", "erf", "multiply", "add", "multiply"});
      return GenerateBody(dense_call, "cutlass_dense_bias_gelu", GetArgumentNames(caller),
                          DenseArgs(std::ref(attrs_)));
    } else if (pattern_name == "cutlass.batch_matmul") {
      const auto* batch_matmul_call =
          GetRootCall(callee->body.as<CallNode>(), 0, std::vector<std::string>{"nn.batch_matmul"});
      return GenerateBody(batch_matmul_call, "cutlass_batch_matmul", GetArgumentNames(caller),
                          BatchMatmulArgs(std::ref(attrs_)));
    } else if (pattern_name == "cutlass.conv2d") {
      const auto* conv2d_call =
          GetRootCall(callee->body.as<CallNode>(), 0, std::vector<std::string>{"nn.conv2d"});
      return GenerateBody(conv2d_call, "cutlass_conv2d", GetArgumentNames(caller),
                          Conv2dArgs(std::ref(attrs_)));
    } else if (pattern_name == "cutlass.conv2d_bias") {
      const CallNode* current_call = callee->body.as<CallNode>();
      std::string add_or_bias_add = current_call->op.as<OpNode>()->name;
      const auto* conv2d_call =
          GetRootCall(callee->body.as<CallNode>(), 1, {"nn.conv2d", add_or_bias_add});
      return GenerateBody(conv2d_call, "cutlass_conv2d_bias", GetArgumentNames(caller),
                          Conv2dArgs(std::ref(attrs_)));
    } else if (pattern_name == "cutlass.conv2d_bias_relu") {
      const CallNode* current_call = callee->body.as<CallNode>();
      std::string add_or_bias_add = current_call->args[0].as<CallNode>()->op.as<OpNode>()->name;
      const auto* conv2d_call =
          GetRootCall(callee->body.as<CallNode>(), 2, {"nn.conv2d", add_or_bias_add, "nn.relu"});
      return GenerateBody(conv2d_call, "cutlass_conv2d_bias_relu", GetArgumentNames(caller),
                          Conv2dArgs(std::ref(attrs_)));
    } else if (pattern_name == "cutlass.conv2d_bias_sigmoid") {
      const CallNode* current_call = callee->body.as<CallNode>();
      std::string add_or_bias_add = current_call->args[0].as<CallNode>()->op.as<OpNode>()->name;
      const auto* conv2d_call =
          GetRootCall(callee->body.as<CallNode>(), 2, {"nn.conv2d", add_or_bias_add, "sigmoid"});
      return GenerateBody(conv2d_call, "cutlass_conv2d_bias_sigmoid", GetArgumentNames(caller),
                          Conv2dArgs(std::ref(attrs_)));
    } else if (pattern_name == "cutlass.conv2d_bias_silu") {
      const CallNode* current_call = callee->body.as<CallNode>();
      std::string add_or_bias_add = current_call->args[0].as<CallNode>()->op.as<OpNode>()->name;
      const auto* conv2d_call =
          GetRootCall(callee->body.as<CallNode>(), 2, {"nn.conv2d", add_or_bias_add, "multiply"});
      return GenerateBody(conv2d_call, "cutlass_conv2d_bias_silu", GetArgumentNames(caller),
                          Conv2dArgs(std::ref(attrs_)));
    } else if (pattern_name == "cutlass.conv2d_bias_hardswish") {
      const CallNode* current_call = callee->body.as<CallNode>();
      std::string add_or_bias_add = current_call->args[0].as<CallNode>()->op.as<OpNode>()->name;
      const auto* conv2d_call =
          GetRootCall(callee->body.as<CallNode>(), 2, {"nn.conv2d", add_or_bias_add, "multiply"});
      return GenerateBody(conv2d_call, "cutlass_conv2d_bias_hardswish", GetArgumentNames(caller),
                          Conv2dArgs(std::ref(attrs_)));
    } else if (IsConv2dResidualBlock(pattern_name.value())) {
      const CallNode* current_call = callee->body.as<CallNode>();
      bool has_relu = current_call->args.size() == 1;
      const CallNode* binop = has_relu ? current_call->args[0].as<CallNode>() : current_call;
      ICHECK(binop->args.size() == 2);
      // Figure out which of the first or second argument corresponds to the residual input
      // The root conv2d call can be reached via the other input of the binary op
      int residual_index;
      if (binop->args[1].as<VarNode>()) {
        residual_index = 1;
      } else if (binop->args[0].as<VarNode>()) {
        residual_index = 0;
      } else {
        const CallNode* lhs = binop->args[0].as<CallNode>();
        const CallNode* rhs = binop->args[1].as<CallNode>();
        ICHECK(lhs && rhs);
        // The residual input should be an ancestor of the non-residual input
        residual_index = IsAncestor(rhs, lhs) ? 1 : 0;
      }
      const auto* non_residual_input = binop->args[!residual_index].as<CallNode>();
      const auto* conv2d_call = GetRootCall(non_residual_input, "nn.conv2d");
      ICHECK(conv2d_call);
      return GenerateBody(conv2d_call, pattern_name.value(), GetArgumentNames(caller),
                          Conv2dArgs(std::ref(attrs_)));
    } else if (pattern_name == "cutlass.conv2d_transpose") {
      const auto* conv2d_call = GetRootCall(callee->body.as<CallNode>(), 0,
                                            std::vector<std::string>{"nn.conv2d_transpose"});
      return GenerateBody(conv2d_call, "cutlass_conv2d_transpose", GetArgumentNames(caller),
                          Conv2dArgs(std::ref(attrs_), true, false));
    } else if (pattern_name == "cutlass.conv2d_backward_weight") {
      const auto* conv2d_call = GetRootCall(callee->body.as<CallNode>(), 0,
                                            std::vector<std::string>{"nn.conv2d_backward_weight"});
      return GenerateBody(conv2d_call, "cutlass_conv2d_backward_weight", GetArgumentNames(caller),
                          Conv2dArgs(std::ref(attrs_), false, true));
    }

    LOG(FATAL) << "Unknown composite function: " << pattern_name;
    return {};
  }

  GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                  const std::vector<std::string>& func_args,
                                  const Str2StrMap& attribute_args) {
    // Make function call with input buffers when visiting arguements
    ICHECK_GT(func_args.size(), 0);
    std::ostringstream decl_stream;
    decl_stream << "(" << func_args[0];
    for (size_t i = 1; i < func_args.size(); ++i) {
      decl_stream << ", " << func_args[i];
    }
    // Analyze the output buffers
    std::vector<Type> out_types;
    if (root_call->checked_type()->IsInstance<TupleTypeNode>()) {
      auto type_node = root_call->checked_type().as<TupleTypeNode>();
      for (auto field : type_node->fields) {
        ICHECK(field->IsInstance<TensorTypeNode>());
        out_types.push_back(field);
      }
    } else if (root_call->checked_type()->IsInstance<TensorTypeNode>()) {
      ICHECK(root_call->checked_type()->IsInstance<TensorTypeNode>());
      out_types.push_back(root_call->checked_type());
    } else {
      LOG(FATAL) << "Unrecognized type node: " << AsText(root_call->checked_type(), false);
    }
    GenerateBodyOutput ret;
    for (const auto& out_type : out_types) {
      const std::string out = "out" + std::to_string(buf_idx_++);
      decl_stream << ", " << out;
      Output output;
      output.name = out;
      output.dtype = GetDtypeString(out_type.as<TensorTypeNode>());
      output.need_copy = false;
      ret.outputs.push_back(output);
    }
    decl_stream << ");";
    if (func_name.find("dense") != std::string::npos) {
      ret.decl = DenseOp(ext_func_id_, attribute_args, func_args);
    } else if (func_name == "cutlass_batch_matmul") {
      ret.decl = BatchMatmulOp(ext_func_id_, attribute_args, func_args);
    } else if (IsConv2dResidualBlock(func_name)) {
      ret.decl = Conv2dOp(ext_func_id_, attribute_args, func_args, true);
    } else if (func_name.find("conv2d") != std::string::npos) {
      ret.decl = Conv2dOp(ext_func_id_, attribute_args, func_args);
    }

    return ret;
  }
  /*! \brief The id of the external cutlass ext_func. */
  std::string ext_func_id_;
  /*! \brief The attrs of the external cutlass ext_func. */
  Map<String, ObjectRef> attrs_;
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls CUTLASS kernels. */
  Array<Var> ext_func_args_;
  /*! \brief Statement of the function that will be compiled using CUTLASS kernels. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The declaration of intermediate buffers. */
  std::vector<std::string> buf_decl_;
};  // class CodegenCutlass

class CutlassModuleCodegen {
 public:
  explicit CutlassModuleCodegen(IRModule mod) : mod_(std::move(mod)) {}

  runtime::Module CreateCSourceModule() {
    EmitPreamble();
    for (const auto& kv : mod_->functions) {
      if (const auto* function_node = GetCutlassFunctionNode(kv.second)) {
        GenCutlassFunc(GetRef<Function>(function_node));
      }
    }
    return Finalize();
  }

 private:
  void EmitPreamble() {
    // create header
    code_stream_ << "#include <cstdint>\n";
    code_stream_ << "#include <cstdlib>\n";
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <vector>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";
    // cutlass header
    code_stream_ << "#include <cuda_fp16.h>\n";
    code_stream_ << "#include <cutlass/cutlass.h>\n";
    code_stream_ << "#include <cutlass/coord.h>\n";
    code_stream_ << "#include <cutlass/tensor_ref.h>\n";
    code_stream_ << "#include <cutlass/util/host_tensor.h>\n";
    code_stream_ << "#include <cutlass/gemm/device/gemm.h>\n";
    code_stream_ << "#include <cutlass/gemm/device/gemm_batched.h>\n";
    code_stream_ << "#include <cutlass/conv/kernel/default_conv2d_fprop.h>\n";
    code_stream_ << "#include <cutlass/conv/kernel/default_conv2d_wgrad.h>\n";
    code_stream_ << "#include <cutlass/conv/kernel/default_conv2d_dgrad.h>\n";
    code_stream_ << "#include <cutlass/conv/device/implicit_gemm_convolution.h>\n";
    code_stream_ << "#include <cutlass/epilogue/thread/linear_combination_bias_relu.h>\n";
    code_stream_ << "#include <cutlass/epilogue/thread/linear_combination_gelu.h>\n";
    code_stream_ << "#include <cutlass/epilogue/thread/linear_combination_sigmoid.h>\n";
    code_stream_ << "#include <cutlass/epilogue/thread/linear_combination_silu.h>\n";
    code_stream_ << "#include <cutlass/epilogue/thread/linear_combination_hardswish.h>\n";
    code_stream_ << "#include <cutlass/epilogue/thread/linear_combination_residual_block.h>\n";
    code_stream_ << "#include <cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h>\n";
    code_stream_ << "#include <cutlass/reduction/device/reduce_split_k.h>\n";
    code_stream_ << "#include <cutlass/reduction/thread/reduction_operators.h>\n";
  }

  void GenCutlassFunc(const Function& function) {
    ICHECK(function.defined()) << "Input error: expect a Relay function.";

    // Record the external symbol for runtime lookup.
    Optional<String> opt_global_symbol = function->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(opt_global_symbol.defined())
        << "CUTLASS functions must have a " << tvm::attr::kGlobalSymbol << " attribute";
    std::string sid = opt_global_symbol.value();
    if (std::find(func_names_.begin(), func_names_.end(), sid) != func_names_.end()) {
      // Already emitted.
      return;
    }
    func_names_.push_back(sid);

    const auto* attrs = function->attrs.as<DictAttrsNode>();
    ICHECK(attrs != nullptr);
    const auto dict = attrs->dict;
    CodegenCutlass builder(sid, dict);
    VLOG(1) << "Creating cutlass C code for '" << sid << "' from:\n" << PrettyPrint(function);
    auto out = builder.VisitExpr(function->body);
    code_stream_ << builder.JIT(out);
  }

  runtime::Module Finalize() {
    ICHECK(!func_names_.empty())
        << "Should only create CUTLASS CSourceModule if have at least one CUTLASS partition";
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    ICHECK(pf != nullptr) << "Cannot find CSource module to create the external runtime module";
    VLOG(1) << "Generated CUTLASS code:" << std::endl << code_stream_.str();
    return (*pf)(code_stream_.str(), "cu", func_names_, /*const_vars=*/Array<String>());
  }

  /*!
   * \brief Returns \p expr as function if it is a \p Function with "Compiler" attribute
   * value "cutlass".
   */
  static const FunctionNode* GetCutlassFunctionNode(const Expr& expr) {
    if (const auto* function_node = expr.as<FunctionNode>()) {
      Optional<String> opt_compiler = function_node->GetAttr<String>(attr::kCompiler);
      if (opt_compiler.defined() && opt_compiler.value() == "cutlass") {
        return function_node;
      }
    }
    return nullptr;
  }

  /*! \brief Module we are compiling. */
  IRModule mod_;
  /*! \brief The accumulated code stream that will be compiled by NVCC */
  std::ostringstream code_stream_;
  /*! \brief The accumulated function names. */
  Array<String> func_names_;
};  // CutlassModuleCodegen

/*!
 * \brief A small shim to redirect to the 'relay.ext.cutlass.compile_for_cutlass' Python
 * function which does the main CUTLASS training, c-code generation and compilation steps.
 */
tvm::transform::Pass CompileForCutlassImpl() {
  auto pass_func = [=](IRModule mod, const tvm::transform::PassContext& pass_ctx) {
    VLOG(1) << "CompileForCutlass input:" << std::endl << PrettyPrint(mod);
    const auto* pf = runtime::Registry::Get("relay.ext.cutlass.compile_for_cutlass");
    ICHECK(pf != nullptr) << "Cannot find compile_for_cutlass function";
    Target target = GetCutlassTarget();
    runtime::Module runtime_mod = (*pf)(mod, target);
    Array<runtime::Module> external_mods =
        mod->GetAttr<Array<runtime::Module>>(tvm::attr::kExternalMods).value_or({});
    external_mods.push_back(runtime_mod);
    return WithAttr(mod, tvm::attr::kExternalMods, external_mods);
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "CompileForCutlass", {});
}

runtime::Module CreateCSourceModule(const IRModule& mod) {
  VLOG(1) << "Creating CUTLASS CSource module from:" << std::endl << PrettyPrint(mod);
  return CutlassModuleCodegen(mod).CreateCSourceModule();
}

}  // namespace

TVM_REGISTER_GLOBAL("relay.ext.cutlass.create_c_source_module").set_body_typed(CreateCSourceModule);

tvm::transform::Pass CompileForCutlass() {
  return transform::Sequential(
      {transform::OutlineCompilerFunctionsWithExistingGlobalSymbols("cutlass"),
       CompileForCutlassImpl(), transform::MarkCompilerFunctionsAsExtern("cutlass")});
}

}  // namespace cutlass
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
