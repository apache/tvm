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

#include "codegen.h"

#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../../transforms/compiler_function_utils.h"
#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cutlass {

std::string EmitSignature(const std::vector<Output>& out, const std::string& func_id,
                          const std::vector<std::string>& arg_names) {
  std::ostringstream code_stream_;
  code_stream_ << "void " << func_id << "_(";
  for (const auto& arg_name : arg_names) {
    code_stream_ << "DLTensor* " << arg_name << ", ";
  }
  for (size_t i = 0; i < out.size() - 1; ++i) {
    code_stream_ << "DLTensor* out" << i << ", ";
  }
  code_stream_ << "DLTensor* out" << out.size() - 1 << ")";
  return code_stream_.str();
}

runtime::Module Finalize(const std::string& code, const Array<String>& func_names) {
  ICHECK(!func_names.empty())
      << "Should only create CUTLASS CSourceModule if there is at least one CUTLASS partition";

  std::ostringstream default_headers;
  default_headers << "#include <tvm/runtime/packed_func.h>\n";
  default_headers << "#include <dlpack/dlpack.h>\n";
  default_headers << "#include <cuda_fp16.h>\n";
  default_headers << "#include <cutlass/cutlass.h>\n";
  default_headers << "#include <cutlass/coord.h>\n";
  default_headers << "#include <cutlass/tensor_ref.h>\n";
  default_headers << "#include <cutlass/util/host_tensor.h>\n";

  const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
  ICHECK(pf != nullptr) << "Cannot find CSource module to create the external runtime module";
  VLOG(1) << "Generated CUTLASS code:" << std::endl << code;
  return (*pf)(default_headers.str() + code, "cu", func_names, /*const_vars=*/Array<String>());
}

class CodegenResultNode : public Object {
 public:
  String code;
  Array<String> headers;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("code", &code);
    v->Visit("headers", &headers);
  }
  static constexpr const char* _type_key = "contrib.cutlass.CodegenResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(CodegenResultNode, Object);
};

class CodegenResult : public ObjectRef {
 public:
  CodegenResult(String code, Array<String> headers) {
    auto n = make_object<CodegenResultNode>();
    n->code = std::move(code);
    n->headers = std::move(headers);
    data_ = std::move(n);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(CodegenResult, ObjectRef, CodegenResultNode)
};

TVM_REGISTER_NODE_TYPE(CodegenResultNode);

TVM_REGISTER_GLOBAL("contrib.cutlass.CodegenResult")
    .set_body_typed([](String code, Array<String> headers) {
      return CodegenResult(code, headers);
    });

GenerateBodyOutput GenerateBody(const std::string& func_name, const std::string& ext_func_id,
                                const std::vector<std::string>& output_types,
                                const Array<String>& func_args, const Map<String, ObjectRef>& attrs,
                                int* buf_idx) {
  // Make function call with input buffers when visiting arguements
  ICHECK_GT(func_args.size(), 0);
  std::ostringstream decl_stream;
  decl_stream << "(" << func_args[0];
  for (size_t i = 1; i < func_args.size(); ++i) {
    decl_stream << ", " << func_args[i];
  }
  GenerateBodyOutput ret;
  for (const auto& out_type : output_types) {
    const std::string out = "out" + std::to_string(*buf_idx++);
    decl_stream << ", " << out;
    Output output;
    output.name = out;
    output.dtype = out_type;
    output.need_copy = false;
    ret.outputs.push_back(output);
  }
  decl_stream << ");";

  const auto* instantiate_template_func =
      runtime::Registry::Get("contrib.cutlass.instantiate_template");
  ICHECK(instantiate_template_func);

  CodegenResult codegen_res = (*instantiate_template_func)(func_name, attrs, func_args);
  ret.decl = codegen_res->code;
  ret.headers = codegen_res->headers;

  return ret;
}

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

class CodegenCutlass : public backend::MemoizedExprTranslator<std::vector<Output>>,
                       public CodegenCBase {
 public:
  CodegenCutlass(const std::string& id, const Map<String, ObjectRef>& attrs) {
    this->ext_func_id_ = id;
    this->attrs_ = attrs;
  }

  std::vector<Output> VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "Cutlass codegen doesn't support: " << op->GetTypeKey();
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
    headers_ = ret.headers;
    return ret.outputs;
  }

  std::string JIT(const std::vector<Output>& out) {
    std::vector<std::string> arg_names;
    for (const auto& arg : ext_func_args_) {
      arg_names.push_back(arg->name_hint());
    }

    code_stream_ << EmitSignature(out, ext_func_id_, arg_names) << "{\n";

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

  Array<String> GetHeaders() { return headers_; }

 private:
  Array<String> GetArgumentNames(const CallNode* call) {
    Array<String> arg_names;
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (const auto& out : res) {
        arg_names.push_back(out.name);
      }
    }
    return arg_names;
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
    const auto pattern_name_opt = callee->GetAttr<runtime::String>(attr::kComposite);
    ICHECK(pattern_name_opt.defined()) << "Only functions with composite attribute are supported.";
    const std::string pattern_name = pattern_name_opt.value();

    if (pattern_name.find("conv2d") != std::string::npos &&
        pattern_name.find("residual") != std::string::npos) {
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
      const auto residual_input = binop->args[residual_index];
      auto call_args = GetArgumentNames(caller);
      auto func_args = call_args;
      if (call_args.size() == 3) {
        // TODO(masahi): This code assumes that there is always a bias_add in a residual block.
        for (size_t i = 0; i < call_args.size(); ++i) {
          if (callee->params[i] == residual_input) {
            auto residual_input_name = call_args[i];
            func_args.push_back(residual_input_name);
          }
        }
      } else {
        ICHECK_EQ(func_args.size(), 4) << "Residual block fusion expects 4 input tensors: data, "
                                          "weight, bias, and residual tensor.";
      }
      return GenerateBody(caller, pattern_name, func_args, attrs_);
    } else {
      return GenerateBody(caller, pattern_name, attrs_);
    }

    LOG(FATAL) << "Unknown composite function: " << pattern_name;
  }

  GenerateBodyOutput GenerateBody(const CallNode* call, const std::string& func_name,
                                  const Array<String>& func_args,
                                  const Map<String, ObjectRef>& attrs) {
    std::vector<Type> out_types;
    if (call->checked_type()->IsInstance<TupleTypeNode>()) {
      auto type_node = call->checked_type().as<TupleTypeNode>();
      for (auto field : type_node->fields) {
        ICHECK(field->IsInstance<TensorTypeNode>());
        out_types.push_back(field);
      }
    } else if (call->checked_type()->IsInstance<TensorTypeNode>()) {
      ICHECK(call->checked_type()->IsInstance<TensorTypeNode>());
      out_types.push_back(call->checked_type());
    } else {
      LOG(FATAL) << "Unrecognized type node: " << AsText(call->checked_type(), false);
    }

    std::vector<std::string> out_types_str;
    for (const auto& out_type : out_types) {
      out_types_str.push_back(GetDtypeString(out_type.as<TensorTypeNode>()));
    }

    return cutlass::GenerateBody(func_name, ext_func_id_, out_types_str, func_args, attrs,
                                 &buf_idx_);
  }

  GenerateBodyOutput GenerateBody(const CallNode* call, const std::string& func_name,
                                  const Map<String, ObjectRef>& attrs) {
    auto func_args = GetArgumentNames(call);
    return GenerateBody(call, func_name, func_args, attrs);
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
  /*! \brief Required header-file names. */
  Array<String> headers_;
};  // class CodegenCutlass

class CutlassModuleCodegen {
 public:
  explicit CutlassModuleCodegen(IRModule mod) : mod_(std::move(mod)) {}

  runtime::Module CreateCSourceModule() {
    for (const auto& entry : mod_->functions) {
      if (const auto* function_node = GetCutlassFunctionNode(entry.second)) {
        GenCutlassFunc(GetRef<Function>(function_node));
      }
    }
    return Finalize(code_stream_.str(), func_names_);
  }

 private:
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
    auto code = builder.JIT(out);
    for (const auto& header : builder.GetHeaders()) {
      code_stream_ << "#include <" << header << ">\n";
    }
    code_stream_ << "\n" + code;
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
