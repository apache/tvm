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
#include <tvm/relay/function.h>

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

TVM_REGISTER_NODE_TYPE(FunctionNode);

TVM_REGISTER_GLOBAL("relay.ir.Function")
    .set_body_typed([](tvm::Array<Var> params, Expr body, Type ret_type,
                       tvm::Array<TypeVar> ty_params, tvm::DictAttrs attrs) {
      return Function(params, body, ret_type, ty_params, attrs);
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
