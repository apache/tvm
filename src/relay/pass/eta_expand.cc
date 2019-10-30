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
 * \file eta_expand.cc
 *
 * \brief Add abstraction over a constructor or global variable bound to a function.
 *
 */
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/relay/expr_functor.h>
#include "../ir/type_functor.h"

namespace tvm {
namespace relay {
namespace eta_expand {

/*!
 * \brief mutator to replace type variables with fresh ones, while maintaining alpha equality
 */
class TypeVarReplacer : public TypeMutator {
 public:
  TypeVarReplacer() : replace_map_({}) {}

  Type VisitType_(const TypeVarNode* type_var_node) final {
    const auto type_var = GetRef<TypeVar>(type_var_node);
    if (replace_map_.find(type_var) == replace_map_.end()) {
      replace_map_[type_var] = TypeVarNode::make("A", Kind::kType);
    }
    return replace_map_[type_var];
  }

 private:
  /*! \brief variable replacement map to remap old type vars to fresh ones */
  std::unordered_map<TypeVar, TypeVar, NodeHash, NodeEqual> replace_map_;
};

/*!
 * \brief mutator to perform eta expansion on all functions in a module
 */
class EtaExpander : public ExprMutator {
 public:
  explicit EtaExpander(
    const Module& mod,
    bool expand_constructor,
    bool expand_global_var)
      : mod_(mod)
      , type_var_replacer_(TypeVarReplacer())
      , expand_constructor_(expand_constructor)
      , expand_global_var_(expand_global_var) {
    CHECK(expand_constructor || expand_global_var)
      << "must expand at least one language feature";
  }

  Module Expand() {
    for (GlobalVar global_var : mod_->GetGlobalVars()) {
      const Function func = mod_->Lookup(global_var);
      const Function new_func = Downcast<Function>(VisitExpr(func));
      mod_->Update(global_var, new_func);
    }
    return mod_;
  }

  Expr VisitExpr_(const CallNode* call) final {
    // we don't need to expand constructors when they are being called, so we
    // prevent them being visited here
    Expr new_op = call->op;
    if (!call->op.as<ConstructorNode>()) {
      new_op = VisitExpr(new_op);
    }
    tvm::Array<Expr> new_args;
    for (const auto& arg : call->args) {
      new_args.push_back(VisitExpr(arg));
    }
    return CallNode::make(new_op, new_args, call->attrs, call->type_args);
  }

  Expr VisitExpr_(const ConstructorNode* cons_node) final {
    Constructor cons = GetRef<Constructor>(cons_node);
    if (!expand_constructor_) {
      return std::move(cons);
    }
    // NOTE: we only reach this case if the constructor is not being applied to any arguments
    tvm::Array<Expr> params;
    for (const auto& type : cons->inputs) {
      Type param_type = type_var_replacer_.VisitType(type);
      params.push_back(VarNode::make("eta_expand_param", param_type));
    }
    tvm::Array<Type> type_params;
    TypeData adt_def = mod_->LookupDef(cons->belong_to);
    for (const auto& type_var : adt_def->type_vars) {
      type_params.push_back(type_var_replacer_.VisitType(type_var));
    }
    Expr body = CallNode::make(cons, params, Attrs());
    Type ret_type = TypeCallNode::make(cons->belong_to, type_params);

    return FunctionNode::make(
      Downcast<tvm::Array<Var>>(params),
      body,
      ret_type,
      Downcast<tvm::Array<TypeVar>>(type_params));
  }

  Expr VisitExpr_(const GlobalVarNode* gvar_node) final {
    GlobalVar gvar = GetRef<GlobalVar>(gvar_node);
    if (!expand_global_var_) {
      return std::move(gvar);
    }

    const auto func = mod_->Lookup(gvar);
    tvm::Array<Expr> params;
    tvm::Array<Var> args;
    for (size_t i = 0; i < func->params.size(); ++i) {
      auto var = VarNode::make("eta_expand_param", func->params[i]->type_annotation);
      params.push_back(var);
      args.push_back(var);
    }

    return FunctionNode::make(
      args,
      CallNode::make(gvar, params),
      func->ret_type,
      func->type_params);
  }

 private:
  /*! \brief reference to module being expanded */
  const Module mod_;
  /*! \brief type variable replacer */
  TypeVarReplacer type_var_replacer_;
  /*! \brief whether to expand constructor nodes */
  bool expand_constructor_;
  /*! \brief whether to expand global variable nodes */
  bool expand_global_var_;
};

}  // namespace eta_expand

namespace transform {

Pass EtaExpand(bool expand_constructor, bool expand_global_var) {
  runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func =
    [=](Module mod, PassContext pc) {
    return eta_expand::EtaExpander(mod, expand_constructor, expand_global_var).Expand();
  };
  return CreateModulePass(pass_func, 1, "EtaExpand", {});
}

TVM_REGISTER_API("relay._transform.EtaExpand")
.set_body_typed(EtaExpand);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
