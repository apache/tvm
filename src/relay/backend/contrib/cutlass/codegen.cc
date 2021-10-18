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

Str2StrMap DenseArgs(const Map<String, ObjectRef>& attrs) {
  Str2StrMap args;
  auto arg0_dtype = std::string(attrs["arg0_dtype"].as<StringObj>()->data);
  auto arg1_dtype = std::string(attrs["arg1_dtype"].as<StringObj>()->data);
  auto ret_dtype = std::string(attrs["ret_dtype"].as<StringObj>()->data);
  auto arg0_shape = attrs["arg0_shape"].as<ArrayNode>();
  auto arg1_shape = attrs["arg1_shape"].as<ArrayNode>();
  args["ElementInputA"] = dtype_map.at(arg0_dtype);
  args["ElementInputB"] = dtype_map.at(arg1_dtype);
  args["ElementOutput"] = dtype_map.at(ret_dtype);
  args["M"] = std::to_string(arg0_shape->at(0).as<IntImmNode>()->value);
  args["K"] = std::to_string(arg0_shape->at(1).as<IntImmNode>()->value);
  args["N"] = std::to_string(arg1_shape->at(0).as<IntImmNode>()->value);
  args["op_def"] = std::string(attrs["cutlass_op_def"].as<StringObj>()->data);
  args["op_name"] = std::string(attrs["cutlass_op_name"].as<StringObj>()->data);
  args["op_type"] = std::string(attrs["op_type"].as<StringObj>()->data);
  args["lda"] = std::string(attrs["lda"].as<StringObj>()->data);
  args["ldb"] = std::string(attrs["ldb"].as<StringObj>()->data);
  args["ldc"] = std::string(attrs["ldc"].as<StringObj>()->data);
  return args;
}

inline void CutlassPrint(std::ostringstream& os, const std::string& stmt, int indent = 2) {
  for (int i = 0; i < indent; ++i) {
    os << " ";
  }
  os << stmt;
}

std::string DenseOp(std::string id, const Str2StrMap& attrs,
                    const std::vector<std::string>& func_args) {
  bool has_bias = false;
  if (attrs.at("op_type") == "cutlass.dense_bias" ||
      attrs.at("op_type") == "cutlass.dense_bias_relu" ||
      attrs.at("op_type") == "cutlass.dense_bias_gelu") {
    has_bias = true;
  }
  std::ostringstream gemm_decl;
  CutlassPrint(gemm_decl, "using ElementInputA = " + attrs.at("ElementInputA") + ";\n");
  CutlassPrint(gemm_decl, "using ElementInputB = " + attrs.at("ElementInputB") + ";\n");
  CutlassPrint(gemm_decl, "using ElementOutput = " + attrs.at("ElementOutput") + ";\n");
  CutlassPrint(gemm_decl, "using ElementComputeEpilogue = " + attrs.at("ElementOutput") + ";\n");
  CutlassPrint(gemm_decl, attrs.at("op_def"));
  CutlassPrint(gemm_decl, "using Gemm = Operation_" + attrs.at("op_name") + ";\n");
  /// Gemm Call

  // Create TensorRef
  CutlassPrint(gemm_decl, "int M = " + attrs.at("M") + ";\n");
  CutlassPrint(gemm_decl, "int N = " + attrs.at("N") + ";\n");
  CutlassPrint(gemm_decl, "int K = " + attrs.at("K") + ";\n");
  CutlassPrint(gemm_decl, "cutlass::gemm::GemmCoord problem_size(M, N, K);\n");
  // Initialize alpha for dot product computation
  CutlassPrint(gemm_decl, "ElementComputeEpilogue alpha = ElementComputeEpilogue(1);\n");
  if (attrs.at("op_type") == "cutlass.dense_bias_gelu") {
    CutlassPrint(gemm_decl, "ElementComputeEpilogue beta = ElementComputeEpilogue(1);\n");
  } else {
    CutlassPrint(gemm_decl, "ElementComputeEpilogue beta = ElementComputeEpilogue(0);\n");
  }

  // Split K dimension into 1 partitions
  CutlassPrint(gemm_decl, "int split_k_slices = 1;\n");

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  ICHECK(func_args.size() >= 2);
  CutlassPrint(gemm_decl, "void* ptr_a = (void*)(" + func_args[0] + ");\n");
  CutlassPrint(gemm_decl, "void* ptr_b = (void*)(" + func_args[1] + ");\n");
  if (has_bias) {
    ICHECK(func_args.size() >= 3);
    CutlassPrint(gemm_decl, "void* ptr_c_bias = (void*)(" + func_args[2] + ");\n");
  }
  CutlassPrint(gemm_decl, "void* ptr_out = (void*)(out0);\n");

  CutlassPrint(gemm_decl, "typename Gemm::Arguments arguments{\n");
  CutlassPrint(gemm_decl, " problem_size,\n");
  CutlassPrint(gemm_decl, " {static_cast<ElementInputA*>(ptr_a), " + attrs.at("lda") + "},\n");
  CutlassPrint(gemm_decl, " {static_cast<ElementInputB*>(ptr_b), " + attrs.at("ldb") + "},\n");
  if (has_bias) {
    CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_c_bias), 0},\n");
  } else {
    CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_out), " + attrs.at("ldc") + "},\n");
  }
  CutlassPrint(gemm_decl, " {static_cast<ElementOutput*>(ptr_out), " + attrs.at("ldc") + "},\n");
  if (has_bias) {
    if (attrs.at("op_type") == "cutlass.dense_bias_gelu") {
      CutlassPrint(gemm_decl, " {alpha, beta},\n");
    } else {
      CutlassPrint(gemm_decl, " {alpha},\n");
    }
  } else {
    CutlassPrint(gemm_decl, " {alpha, beta},\n");
  }
  CutlassPrint(gemm_decl, " split_k_slices};\n");

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  CutlassPrint(gemm_decl, "size_t workspace_size = Gemm::get_workspace_size(arguments);\n");
  // Allocate workspace memory
  CutlassPrint(gemm_decl,
               "cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);\n");
  // Instantiate CUTLASS kernel depending on template
  CutlassPrint(gemm_decl, "Gemm gemm_op;\n");
  // Check the problem size is supported or not
  CutlassPrint(gemm_decl, "cutlass::Status status = gemm_op.can_implement(arguments);\n");
  CutlassPrint(gemm_decl, "CHECK(status == cutlass::Status::kSuccess);\n");
  // Initialize CUTLASS kernel with arguments and workspace pointer
  CutlassPrint(gemm_decl, "status = gemm_op.initialize(arguments, workspace.get());\n");
  CutlassPrint(gemm_decl, "CHECK(status == cutlass::Status::kSuccess);\n");
  // Launch initialized CUTLASS kernel
  CutlassPrint(gemm_decl, "status = gemm_op();\n");
  CutlassPrint(gemm_decl, "CHECK(status == cutlass::Status::kSuccess);\n");
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
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
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
    } else if (pattern_name == "cutlass.dense_bias_gelu") {
      const CallNode* current_call = callee->body.as<CallNode>();
      std::string add_or_bias_add = current_call->args[1].as<CallNode>()->op.as<OpNode>()->name;
      const auto* dense_call = GetRootCall(callee->body.as<CallNode>(), 8,
                                           {"nn.dense", add_or_bias_add, "multiply", "cast", "erf",
                                            "cast", "multiply", "add", "multiply"});
      return GenerateBody(dense_call, "cutlass_dense_bias_gelu", GetArgumentNames(caller),
                          DenseArgs(std::ref(attrs_)));
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
