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
 * \file src/relax/backend/contrib/cutlass/codegen.cc
 * \brief Implementation of the CUTLASS code generator for Relax.
 */
#include "../../../../relay/backend/contrib/cutlass/codegen.h"

#include <tvm/ir/module.h>
#include <tvm/ir/name_supply.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/type.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../../relay/backend/contrib/codegen_c/codegen_c.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

using namespace relay::contrib::cutlass;

using Output = relay::contrib::Output;
using GenerateBodyOutput = relay::contrib::GenerateBodyOutput;
using relay::contrib::cutlass::GenerateBody;
using OutputType = std::vector<Output>;

class CodegenCutlass : public relax::MemoizedExprTranslator<OutputType>,
                       public relay::contrib::CodegenCBase {
 public:
  CodegenCutlass(const std::string& id, const Map<Var, Expr>& bindings)
      : ext_func_id_(id), bindings_(bindings), name_sup_("") {}

  void AddParm(Var param) {
    ext_func_args_.push_back(param);
    auto v_name = name_sup_->FreshName(param->name_hint());
    var_name_map_[param.get()] = v_name;
  }

  std::string JIT(const OutputType& out) final {
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

    code_stream_ << EmitSignature(out, ext_func_id_, arg_names) << "{\n";

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
  OutputType VisitExpr_(const VarNode* node) final {
    Output output;
    auto it = var_name_map_.find(node);
    ICHECK(it != var_name_map_.end());
    output.name = it->second;
    return {output};
  }

  OutputType VisitExpr_(const CallNode* call) final {
    const auto* fn_var = call->op.as<VarNode>();
    ICHECK(fn_var);
    const auto func = Downcast<Function>(bindings_[GetRef<Var>(fn_var)]);
    const auto pattern_name_opt = func->GetAttr<runtime::String>(attr::kComposite);
    ICHECK(pattern_name_opt) << "Only composite function is supported for CUTLASS.";
    auto ret = GenerateBody(call, pattern_name_opt.value(), func->attrs->dict);
    ext_func_body_.push_back(ret.decl);
    headers_ = ret.headers;
    return ret.outputs;
  }

  OutputType VisitExpr_(const FunctionNode* fn) {
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

  OutputType VisitExpr_(const SeqExprNode* op) {
    OutputType outputs;

    for (BindingBlock block : op->blocks) {
      VisitBindingBlock(block);
    }

    auto from_body = VisitExpr(op->body);
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

  GenerateBodyOutput GenerateBody(const CallNode* call, const std::string& func_name,
                                  const Map<String, ObjectRef>& attrs) {
    auto func_args = GetArgumentNames(call);
    auto struct_info = GetStructInfo(GetRef<Call>(call));

    std::vector<std::string> out_types;
    if (const auto* tensor_sinfo = struct_info.as<TensorStructInfoNode>()) {
      out_types.emplace_back(backend::DType2String(tensor_sinfo->dtype));
    } else {
      LOG(FATAL) << "Unimplemented sinfo type: " << struct_info;
    }

    return contrib::GenerateBody(func_name, ext_func_id_, out_types, func_args, attrs, &buf_idx_);
  }

  /*! \brief The id of the external cutlass ext_func. */
  std::string ext_func_id_;
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls CUTLASS kernels. */
  Array<Var> ext_func_args_;
  /*! \brief The statements of the function that will be compiled using CUTLASS kernels. */
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

class CutlassModuleCodegen {
 public:
  runtime::Module CreateCSourceModule(Array<Function> functions,
                                      const Map<String, ObjectRef>& options) {
    std::string headers = "";
    std::string code = "";
    for (const auto& f : functions) {
      auto [f_code, op_headers] = GenCutlassFunc(f, options);
      code += "\n" + f_code;
      for (const auto& header : op_headers) {
        headers += "#include <" + header + ">\n";
      }
    }
    return Finalize(headers + "\n" + code, func_names_);
  }

 private:
  std::pair<std::string, Array<String>> GenCutlassFunc(const Function& function,
                                                       const Map<String, ObjectRef>& options) {
    ICHECK(function.defined()) << "Input error: expect a Relay function.";

    auto sid = GetExtSymbol(function);
    func_names_.push_back(sid);

    CodegenCutlass builder(sid, AnalyzeVar2Value(function));

    for (const auto& p : function->params) {
      builder.AddParm(p);
    }

    auto out = builder.VisitExpr(function->body);
    return {builder.JIT(out), builder.GetHeaders()};
  }

  /*! \brief The accumulated function names. */
  Array<String> func_names_;
};

Array<runtime::Module> CUTLASSCompiler(Array<Function> functions, Map<String, ObjectRef> options,
                                       Map<Constant, String> /*unused*/) {
  const auto* tune_func = runtime::Registry::Get("contrib.cutlass.tune_relax_function");
  ICHECK(tune_func != nullptr)
      << "The packed function contrib.cutlass.tune_relax_function not found, "
         "please import tvm.contrib.cutlass.build";

  Array<Function> annotated_functions = (*tune_func)(functions, options);

  auto source_mod = CutlassModuleCodegen().CreateCSourceModule(annotated_functions, options);
  const auto* pf = runtime::Registry::Get("contrib.cutlass.compile");
  ICHECK(pf != nullptr) << "The packed function contrib.cutlass.compile not found, please import "
                           "tvm.contrib.cutlass.build";
  runtime::Module cutlass_mod = (*pf)(source_mod, options);

  return {cutlass_mod};
}

TVM_REGISTER_GLOBAL("relax.ext.cutlass").set_body_typed(CUTLASSCompiler);

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
