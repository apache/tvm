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
 * \file src/relax/backend/contrib/composable_kernel/codegen.cc
 * \brief Implementation of the Composable Kernel code generator for Relax.
 */
#include <tvm/ir/module.h>
#include <tvm/ir/name_supply.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/type.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../../relay/backend/contrib/codegen_c/codegen_c.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

using Output = relay::contrib::Output;
using GenerateBodyOutput = relay::contrib::GenerateBodyOutput;
using OutputType = std::vector<Output>;

class CodegenResultNode : public Object {
 public:
  String code;
  Array<String> headers;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("code", &code);
    v->Visit("headers", &headers);
  }
  static constexpr const char* _type_key = "contrib.composable_kernel.CodegenResult";
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

TVM_REGISTER_GLOBAL("contrib.composable_kernel.CodegenResult")
    .set_body_typed([](String code, Array<String> headers) {
      return CodegenResult(code, headers);
    });

class CodegenComposableKernel : public relax::MemoizedExprTranslator<OutputType>,
                                public relay::contrib::CodegenCBase {
 public:
  CodegenComposableKernel(const std::string& id, const Map<Var, Expr>& bindings)
      : ext_func_id_(id), bindings_(bindings), name_sup_("") {}

  void AddParam(Var param) {
    ext_func_args_.push_back(param);
    auto v_name = name_sup_->FreshName(param->name_hint());
    var_name_map_[param.get()] = v_name;
  }

  std::string JIT(const std::vector<Output>& out) final {
    std::vector<std::string> arg_types, arg_names;

    for (const auto& arg : ext_func_args_) {
      auto sinfo = GetStructInfo(arg);
      if (const auto* tensor_sinfo = sinfo.as<TensorStructInfoNode>()) {
        arg_types.emplace_back(backend::DType2String(tensor_sinfo->dtype));
      } else if (const auto* shape_sinfo = sinfo.as<ShapeStructInfoNode>()) {
        arg_types.emplace_back(backend::DType2String(shape_sinfo->values.value()[0]->dtype));
      } else {
        LOG(FATAL) << "Unimplemented";
      }
      arg_names.push_back(var_name_map_.at(arg.get()));
    }

    EmitSignature(out, ext_func_id_, arg_names);
    code_stream_ << " {\n";

    this->EnterScope();

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

    this->GenerateBackendCFunc(ext_func_id_, arg_types, /*const_arr_name=*/"", out, true);

    return code_stream_.str();
  }

  Array<String> GetHeaders() { return headers_; }

 protected:
  OutputType VisitExpr_(const VarNode* var) final {
    Output output;
    auto it = var_name_map_.find(var);
    ICHECK(it != var_name_map_.end());
    output.name = it->second;
    return {output};
  }

  OutputType VisitExpr_(const CallNode* call) final {
    const auto* fn_var = call->op.as<VarNode>();
    ICHECK(fn_var);
    const auto func = Downcast<Function>(bindings_[GetRef<Var>(fn_var)]);
    const auto pattern_name_opt = func->GetAttr<runtime::String>(attr::kComposite);
    ICHECK(pattern_name_opt) << "Only composite function is supported for ComposableKernel.";
    auto ret = GenerateBody(call, func);
    ext_func_body_.push_back(ret.decl);
    headers_ = ret.headers;
    return ret.outputs;
  }

  OutputType VisitExpr_(const FunctionNode* fn) final {
    ICHECK(fn->GetAttr<String>(attr::kComposite).defined())
        << "JSON runtime only supports composite functions";
    // FunctionNode should be handled by the caller.
    return {};
  }

  OutputType VisitBinding(const Binding& binding) {
    OutputType outputs;
    if (const auto* node = binding.as<VarBindingNode>()) {
      auto from_b = VisitBinding_(node);
      outputs.insert(outputs.end(), from_b.begin(), from_b.end());
    } else {
      LOG(FATAL) << "Unimplemented type: " << binding->GetTypeKey();
    }
    return outputs;
  }

  OutputType VisitBindingBlock(const BindingBlock& block) {
    OutputType outputs;
    if (const auto* node = block.as<DataflowBlockNode>()) {
      auto from_bb = VisitBindingBlock_(node);
      outputs.insert(outputs.end(), from_bb.begin(), from_bb.end());
    } else if (const auto* node = block.as<BindingBlockNode>()) {
      auto from_bb = VisitBindingBlock_(node);
      outputs.insert(outputs.end(), from_bb.begin(), from_bb.end());
    } else {
      LOG(FATAL) << "Unimplemented type: " << block->GetTypeKey();
    }
    return outputs;
  }

  OutputType VisitBindingBlock_(const BindingBlockNode* block) {
    OutputType outputs;
    for (Binding binding : block->bindings) {
      auto from_b = VisitBinding(binding);
      outputs.insert(outputs.end(), from_b.begin(), from_b.end());
    }
    return outputs;
  }

  OutputType VisitBindingBlock_(const DataflowBlockNode* block) {
    OutputType outputs;
    for (Binding binding : block->bindings) {
      auto from_b = VisitBinding(binding);
      outputs.insert(outputs.end(), from_b.begin(), from_b.end());
    }
    return outputs;
  }

  OutputType VisitExpr_(const SeqExprNode* seqExpr) final {
    OutputType outputs;

    for (BindingBlock block : seqExpr->blocks) {
      VisitBindingBlock(block);
    }

    auto from_body = VisitExpr(seqExpr->body);
    outputs.insert(outputs.end(), from_body.begin(), from_body.end());

    return outputs;
  }

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

  void EmitSignature(const std::vector<Output>& out, const std::string& func_id,
                     const std::vector<std::string>& arg_names) {
    code_stream_ << "void " << func_id << "_(";
    for (const auto& arg_name : arg_names) {
      code_stream_ << "DLTensor* " << arg_name << ", ";
    }
    for (size_t i = 0; i < out.size() - 1; ++i) {
      code_stream_ << "DLTensor* out" << i << ", ";
    }
    code_stream_ << "DLTensor* out" << out.size() - 1 << ")";
  }

  GenerateBodyOutput GenerateBody(const CallNode* call, const Function func) {
    auto func_args = GetArgumentNames(call);
    auto struct_info = GetStructInfo(GetRef<Call>(call));
    std::vector<std::string> out_types;
    if (const auto* tensor_sinfo = struct_info.as<TensorStructInfoNode>()) {
      out_types.emplace_back(backend::DType2String(tensor_sinfo->dtype));
    } else {
      LOG(FATAL) << "Unimplemented sinfo type: " << struct_info;
    }

    ICHECK_GT(func_args.size(), 0);
    std::ostringstream decl_stream;
    decl_stream << "(" << func_args[0];
    for (size_t i = 1; i < func_args.size(); ++i) {
      decl_stream << ", " << func_args[i];
    }

    GenerateBodyOutput ret;
    for (const auto& out_type : out_types) {
      const std::string out = "out" + std::to_string(buf_idx_++);
      decl_stream << ", " << out;
      Output output;
      output.name = out;
      output.dtype = out_type;
      output.need_copy = false;
      ret.outputs.push_back(output);
    }
    decl_stream << ");";

    const auto* generate_kernel_func =
        runtime::Registry::Get("contrib.composable_kernel.generate_kernel");
    ICHECK(generate_kernel_func);

    CodegenResult codegen_res = (*generate_kernel_func)(GetRef<Call>(call), func);
    ret.decl = codegen_res->code;
    ret.headers = codegen_res->headers;

    return ret;
  }

 private:
  /*! \brief The id of the external ComposableKernel ext_func. */
  std::string ext_func_id_;
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls ComposableKernel kernels. */
  Array<Var> ext_func_args_;
  /*! \brief The statements of the function that will be compiled using ComposableKernel kernels. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The declaration of intermediate buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The binding to look up composite functions. */
  Map<Var, Expr> bindings_;
  /*! \brief Required header-file names. */
  Array<String> headers_;
  /*!
   * \brief A mapping from a variable to its unique name.
   * We use this since sometimes different parameters to the same function end up having the same
   * name_hint.
   */
  std::unordered_map<const VarNode*, std::string> var_name_map_;
  /*! \brief A name supply to generate a unique name for each parameter. */
  NameSupply name_sup_;
};

class ComposableKernelModuleCodegen {
 public:
  runtime::Module CreateCSourceModule(Array<Function> functions,
                                      const Map<String, ObjectRef>& options) {
    std::string headers = "";
    std::string code = "";
    for (const auto& f : functions) {
      auto [f_code, op_headers] = GenComposableKernelFunc(f, options);
      code += "\n" + f_code;
      for (const auto& header : op_headers) {
        headers += "#include <" + header + ">\n";
      }
    }
    return Finalize(headers + "\n" + code, func_names_);
  }

 private:
  std::pair<std::string, Array<String>> GenComposableKernelFunc(
      const Function& function, const Map<String, ObjectRef>& options) {
    ICHECK(function.defined()) << "Input error: expect a Relay function.";

    auto sid = GetExtSymbol(function);
    func_names_.push_back(sid);

    CodegenComposableKernel builder(sid, AnalyzeVar2Value(function));

    for (const auto& p : function->params) {
      builder.AddParam(p);
    }

    auto out = builder.VisitExpr(function->body);
    return {builder.JIT(out), builder.GetHeaders()};
  }

  runtime::Module Finalize(const std::string& code, const Array<String>& func_names) {
    ICHECK(!func_names.empty()) << "Should only create ComposableKernel CSourceModule if there is "
                                   "at least one ComposableKernel partition";

    std::ostringstream default_headers;
    default_headers << "#include <tvm/runtime/packed_func.h>\n";
    default_headers << "#include <dlpack/dlpack.h>\n";

    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    ICHECK(pf != nullptr) << "Cannot find CSource module to create the external runtime module";
    return (*pf)(default_headers.str() + code, "cpp", func_names, /*const_vars=*/Array<String>());
  }

  /*! \brief The accumulated function names. */
  Array<String> func_names_;
};

Array<runtime::Module> ComposableKernelCompiler(Array<Function> functions,
                                                Map<String, ObjectRef> options,
                                                Map<Constant, String> /*unused*/) {
  auto source_mod = ComposableKernelModuleCodegen().CreateCSourceModule(functions, options);

  const auto* pf = runtime::Registry::Get("contrib.composable_kernel.compile");
  ICHECK(pf != nullptr)
      << "The packed function contrib.composable_kernel.compile not found, please import "
         "tvm.contrib.composable_kernel.build";
  runtime::Module composable_kernel_mod = (*pf)(source_mod, options);

  return {composable_kernel_mod};
}

TVM_REGISTER_GLOBAL("relax.ext.composable_kernel").set_body_typed(ComposableKernelCompiler);

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
