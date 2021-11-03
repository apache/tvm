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
 * \brief Implementation of CUTLASS codegen.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <numeric>
#include <sstream>

#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;
using Str2StrMap = std::unordered_map<std::string, std::string>;

static Str2StrMap dtype_map = {{"float16", "cutlass::half_t"}, {"float32", "float"}};

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

Str2StrMap GemmArgsCommon(const Map<String, ObjectRef>& attrs) {
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
  CutlassPrint(gemm_decl, "using Gemm = Operation_" + attrs.at("op_name") + ";\n");

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

  CutlassPrint(gemm_decl, "void* ptr_out = (void*)(out0);\n");

  CutlassPrint(gemm_decl, "using " + kernel + " = Operation_" + attrs.at("op_name") + ";\n");
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
  bool has_bias = false;
  bool is_gelu =
      attrs.at("op_type").find("cutlass.dense_bias_gelu") != std::string::npos;  // fp32 or fp16
  if (attrs.at("op_type") == "cutlass.dense_bias" ||
      attrs.at("op_type") == "cutlass.dense_bias_relu" || is_gelu) {
    has_bias = true;
  }
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

  // "batch_stride_A": arg0_shape[1] * arg0_shape[2],
  // "batch_stride_B": arg1_shape[1] * arg1_shape[2],
  // "batch_stride_C": arg0_shape[1] * arg1_shape[1],

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

class CodegenCutlass : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
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
      code_stream_ << out[i].dtype << "* out" << i << ", ";
    }
    code_stream_ << out.back().dtype << "* out" << out.size() - 1 << ") {\n";
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

    this->GenerateBackendCFunc(ext_func_id_, ext_func_args_, const_array_name_, out, true);
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

  GenerateBodyOutput GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                   const CallNode* caller) {
    const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
    ICHECK(pattern_name.defined()) << "Only functions with composite attribute are supported.";

    if (pattern_name == "cutlass.dense") {
      const auto* dense_call = GetRootCall(callee->body.as<CallNode>(), 0, {"nn.dense"});
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
          GetRootCall(callee->body.as<CallNode>(), 0, {"nn.batch_matmul"});
      return GenerateBody(batch_matmul_call, "cutlass_batch_matmul", GetArgumentNames(caller),
                          BatchMatmulArgs(std::ref(attrs_)));
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
    if (func_name == "cutlass_dense" || func_name == "cutlass_dense_bias" ||
        func_name == "cutlass_dense_bias_relu" || func_name == "cutlass_dense_bias_gelu") {
      ret.decl = DenseOp(ext_func_id_, attribute_args, func_args);
    } else if (func_name == "cutlass_batch_matmul") {
      ret.decl = BatchMatmulOp(ext_func_id_, attribute_args, func_args);
    }
    return ret;
  }
  /*! \brief The id of the external cutlass ext_func. */
  std::string ext_func_id_{""};
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
  /*! \brief The array declared to store the constant values. */
  std::string const_array_name_;
  /*! \brief The declaration of intermediate buffers. */
  std::vector<std::string> buf_decl_;
};  // class CodegenCutlass

class CutlassModuleCodegen : public CSourceModuleCodegenBase {
 public:
  std::pair<std::string, Array<String>> GenCutlassFunc(const Function& func) {
    ICHECK(func.defined()) << "Input error: expect a Relay function.";
    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);
    const auto* attrs = func->attrs.as<DictAttrsNode>();
    ICHECK(attrs != nullptr);
    const auto dict = attrs->dict;
    CodegenCutlass builder(sid, dict);
    auto out = builder.VisitExpr(func->body);
    code_stream_ << builder.JIT(out);
    return {sid, {}};
  }

  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
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
    code_stream_ << "#include <cutlass/util/host_tensor.h>\n";
    code_stream_ << "#include <cutlass/util/reference/host/tensor_fill.h>\n";
    code_stream_ << "#include <cutlass/gemm/device/gemm.h>\n";
    code_stream_ << "#include <cutlass/gemm/device/gemm_batched.h>\n";
    code_stream_ << "#include <cutlass/epilogue/thread/linear_combination_bias_relu.h>\n";
    code_stream_ << "#include <cutlass/epilogue/thread/linear_combination_gelu.h>\n";

    ICHECK(ref->IsInstance<FunctionNode>());
    auto res = GenCutlassFunc(Downcast<Function>(ref));
    std::string code = code_stream_.str();
    String sym = std::get<0>(res);
    Array<String> variables = std::get<1>(res);
    // Create a CSource module
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    ICHECK(pf != nullptr) << "Cannot find CSource module to create the external runtime module";
    return (*pf)(code, "cu", Array<String>{sym}, variables);
  }

 private:
  /*! \brief The code stream that will be compiled by NVCC */
  std::ostringstream code_stream_;
};  // CutlassModuleCodegen

/*!
 * \brief The external cutlass compiler/codegen tool. It takes a Relay
 * expression/module and compile it into a runtime module.
 */
runtime::Module CutlassCompiler(const ObjectRef& ref) {
  CutlassModuleCodegen cutlass;
  return cutlass.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.cutlass").set_body_typed(CutlassCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
