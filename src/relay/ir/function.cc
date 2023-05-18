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
 * \file src/relay/ir/function.cc
 * \brief Function in relay.
 */
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/function.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

Function::Function(tvm::Array<Var> params, Expr body, Type ret_type,
                   tvm::Array<TypeVar> type_params, DictAttrs attrs, Span span) {
  ObjectPtr<FunctionNode> n = make_object<FunctionNode>();
  ICHECK(params.defined());
  ICHECK(type_params.defined());
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->type_params = std::move(type_params);
  n->attrs = std::move(attrs);
  n->virtual_device_ = VirtualDevice::FullyUnconstrained();
  n->span = std::move(span);
  data_ = std::move(n);
}

Function WithFields(Function function, Optional<Array<Var>> opt_params, Optional<Expr> opt_body,
                    Optional<Type> opt_ret_type, Optional<Array<TypeVar>> opt_ty_params,
                    Optional<DictAttrs> opt_attrs, Optional<VirtualDevice> opt_virtual_device,
                    Optional<Span> opt_span) {
  Array<Var> params = opt_params.value_or(function->params);
  Expr body = opt_body.value_or(function->body);
  Type ret_type = opt_ret_type.value_or(function->ret_type);
  Array<TypeVar> ty_params = opt_ty_params.value_or(function->type_params);
  DictAttrs attrs = opt_attrs.value_or(function->attrs);
  VirtualDevice virtual_device = opt_virtual_device.value_or(function->virtual_device());
  Span span = opt_span.value_or(function->span);

  bool unchanged = body.same_as(function->body) && ret_type.same_as(function->ret_type) &&
                   attrs.same_as(function->attrs) &&
                   virtual_device.same_as(function->virtual_device()) &&
                   span.same_as(function->span);

  // Check that all the type params are unchanged
  if (unchanged) {
    bool all_ty_params_unchanged = true;
    if (ty_params.size() == function->type_params.size()) {
      for (size_t i = 0; i < ty_params.size(); i++) {
        all_ty_params_unchanged &= ty_params[i].same_as(function->type_params[i]);
      }
    } else {
      all_ty_params_unchanged = false;
    }
    unchanged &= all_ty_params_unchanged;
  }

  // Check that all the params are unchanged
  if (unchanged) {
    bool all_params_unchanged = true;
    if (params.size() == function->params.size()) {
      for (size_t i = 0; i < params.size(); i++) {
        all_params_unchanged &= params[i].same_as(function->params[i]);
      }
    } else {
      all_params_unchanged = false;
    }
    unchanged &= all_params_unchanged;
  }

  if (!unchanged) {
    FunctionNode* cow_function_node = function.CopyOnWrite();
    cow_function_node->params = params;
    cow_function_node->body = body;
    cow_function_node->ret_type = ret_type;
    cow_function_node->type_params = ty_params;
    cow_function_node->attrs = attrs;
    cow_function_node->virtual_device_ = virtual_device;
    cow_function_node->span = span;
  }
  return function;
}

FuncType FunctionNode::func_type_annotation() const {
  Array<Type> param_types;
  for (auto param : this->params) {
    Type param_type =
        (param->type_annotation.defined()) ? param->type_annotation : IncompleteType(Kind::kType);
    param_types.push_back(param_type);
  }

  Type ret_type = (this->ret_type.defined()) ? this->ret_type : IncompleteType(Kind::kType);
  return FuncType(param_types, ret_type, this->type_params, {});
}

const FunctionNode* AsOptimizableFunctionNode(const BaseFunc& base_func) {
  if (const auto* function_node = base_func.as<FunctionNode>()) {
    if (!function_node->GetAttr<String>(attr::kCompiler).defined() &&
        !function_node->HasNonzeroAttr(attr::kExtern) &&
        !function_node->HasNonzeroAttr(attr::kSkipOptimization)) {
      return function_node;
    }
  }
  return nullptr;
}

TVM_REGISTER_GLOBAL("relay.ir.PrintRelayModule")
    .set_body_typed([](IRModule mod) -> Optional<String> {
      for (const auto& it : mod->functions) {
        if (it.second->IsInstance<FunctionNode>()) {
          return PrettyPrint(mod);
        }
      }
      return NullOpt;
    });

TVM_REGISTER_GLOBAL("relay.ir.PrintIR")
    .set_body_typed([](IRModule mod, String header, bool show_metadata) -> bool {
      for (const auto& it : mod->functions) {
        if (it.second->IsInstance<FunctionNode>()) {
          LOG(INFO) << "PrintIR(" << header << "):\n" << AsText(mod, show_metadata);
          return true;
        }
      }
      return false;
    });

TVM_REGISTER_GLOBAL("relay.ir.WarnIfMalformed")
    .set_body_typed([](const IRModule& mod, const BaseFunc& base_func) -> void {
      if (auto relay_func = base_func.as<Function>()) {
        Function func = Downcast<relay::Function>(relay::DeDup(relay_func.value()));
        // Type check the item before we add it to the module.
        auto fv = relay::FreeVars(func);
        auto ftv = relay::FreeTypeVars(func, mod);
        // TODO(@jroesch): refactor to use diagnostic context
        ICHECK_EQ(fv.size(), 0) << "Function:" << std::endl
                                << PrettyPrint(func) << std::endl
                                << "contains free variables: " << fv;
        ICHECK_EQ(ftv.size(), 0) << "Function:" << std::endl
                                 << PrettyPrint(func) << std::endl
                                 << "contains free type variables: " << fv;
      }
    });
TVM_REGISTER_GLOBAL("relay.ir.IRModuleAdd")
    .set_body_typed([](IRModule mod, GlobalVar var, ObjectRef val, bool update) -> IRModule {
      if (val->IsInstance<BaseFuncNode>()) {
        mod->Add(var, Downcast<BaseFunc>(val), update);
      } else if (val->IsInstance<GlobalVarNode>()) {
        GlobalVar gv = Downcast<GlobalVar>(val);
        IRModule mod_copy(make_object<IRModuleNode>(*mod.operator->()));
        mod_copy = relay::transform::EtaExpand(
            /* expand_constructor */ false,
            /* expand_global_var */ true)(mod_copy);
        auto func = mod_copy->Lookup(gv->name_hint);
        mod->Add(var, Downcast<relay::Function>(func), update);
      } else {
        auto func = relay::Function({}, Downcast<RelayExpr>(val), Type(nullptr), {});
        mod->Add(var, func, update);
      }
      return mod;
    });

TVM_REGISTER_GLOBAL("relay.ir.IRModuleUpdateWithRenamer")
    .set_body_typed([](IRModule self, IRModule mod) -> void {
      struct Renamer : relay::ExprMutator, TypeMutator {
        Map<String, GlobalVar> defs;
        Map<String, GlobalTypeVar> types;
        std::unordered_map<int32_t, Constructor> ctors;

        Renamer(Map<String, GlobalVar> defs_one, Map<String, GlobalVar> defs_two,
                Map<String, GlobalTypeVar> types_one, Map<String, GlobalTypeVar> types_two,
                std::unordered_map<int32_t, Constructor> ctors_one,
                std::unordered_map<int32_t, Constructor> ctor_two) {
          for (auto pair : defs_one) {
            defs.Set(pair.first, pair.second);
          }

          for (auto pair : defs_two) {
            auto it = defs.find(pair.first);
            if (it == defs.end()) {
              defs.Set(pair.first, pair.second);
            }
          }

          for (auto pair : types_one) {
            types.Set(pair.first, pair.second);
          }

          for (auto pair : types_two) {
            auto it = types.find(pair.first);
            if (it == types.end()) {
              types.Set(pair.first, pair.second);
            }
          }
        }

        relay::Expr VisitExpr_(const GlobalVarNode* node) override {
          return defs.at(node->name_hint);
        }

        Type VisitType_(const GlobalTypeVarNode* node) override {
          return types.at(node->name_hint);
        }
      };

      Renamer renamer(self->global_var_map_, mod->global_var_map_, self->global_type_var_map_,
                      mod->global_type_var_map_, self->constructor_tag_map_,
                      mod->constructor_tag_map_);

      self->global_var_map_ = renamer.defs;
      self->global_type_var_map_ = renamer.types;
      self->constructor_tag_map_ = renamer.ctors;

      for (auto pair : mod->type_definitions) {
        auto tvar = renamer.types.at(pair.first->name_hint);
        auto ty = renamer.ExprMutator::VisitType(pair.second);
        self->AddTypeDefUnchecked(tvar, Downcast<TypeData>(ty), true);
      }

      for (auto pair : mod->functions) {
        if (auto rfn = pair.second.as<relay::FunctionNode>()) {
          auto gvar = renamer.defs.at(pair.first->name_hint);
          auto fn = renamer.VisitExpr(GetRef<relay::Function>(rfn));
          self->AddUnchecked(gvar, Downcast<BaseFunc>(fn));
        } else {
          // TODO(@jroesch): rename into IRModule.
          self->AddUnchecked(pair.first, pair.second);
        }
      }
    });

TVM_REGISTER_GLOBAL("relay.ir.FunctionFromExprInContext")
    .set_body_typed([](RelayExpr expr, IRModule mod) -> Function {
      return Function(relay::FreeVars(expr), expr, Type(), relay::FreeTypeVars(expr, mod), {});
    });

TVM_REGISTER_GLOBAL("relay.ir.FuncWithAttr")
    .set_body_typed([](BaseFunc func, String key, ObjectRef value) -> Optional<Function> {
      if (func->IsInstance<relay::FunctionNode>()) {
        return WithAttr(Downcast<relay::Function>(std::move(func)), key, value);
      }
      return NullOpt;
    });

TVM_REGISTER_GLOBAL("relay.ir.FuncWithoutAttr")
    .set_body_typed([](BaseFunc func, String key) -> Optional<Function> {
      if (func->IsInstance<relay::FunctionNode>()) {
        return WithoutAttr(Downcast<relay::Function>(std::move(func)), key);
      }
      return NullOpt;
    });

TVM_REGISTER_NODE_TYPE(FunctionNode);

TVM_REGISTER_GLOBAL("relay.ir.Function")
    .set_body_typed([](tvm::Array<Var> params, Expr body, Type ret_type,
                       tvm::Array<TypeVar> ty_params, tvm::DictAttrs attrs, Span span) {
      return Function(params, body, ret_type, ty_params, attrs, span);
    });
TVM_REGISTER_GLOBAL("relay.ir.FunctionWithFields")
    .set_body_typed([](Function function, Optional<Array<Var>> opt_params, Optional<Expr> opt_body,
                       Optional<Type> opt_ret_type, Optional<Array<TypeVar>> opt_ty_params,
                       Optional<DictAttrs> opt_attrs, Optional<VirtualDevice> opt_virtual_device,
                       Optional<Span> opt_span) {
      return WithFields(function, opt_params, opt_body, opt_ret_type, opt_ty_params, opt_attrs,
                        opt_virtual_device, opt_span);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FunctionNode>([](const ObjectRef& ref, ReprPrinter* p) {
      // TODO(@jroesch): previously this had a debug printer, the debug printer
      // can cause exponential behavior and is currently dangerous, for these
      // cases we need some kind of de-duping.
      //
      // See old implementation:
      //
      // auto* node = static_cast<const FunctionNode*>(ref.get());
      // p->stream << "FunctionNode(" << node->params << ", " << node->ret_type << ", " <<
      // node->body
      //           << ", " << node->type_params << ", " << node->attrs << ")";
      p->stream << PrettyPrint(ref);
    });

}  // namespace relay
}  // namespace tvm
