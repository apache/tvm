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
 * \file src/relay/transforms/device_aware_visitors.cc
 * \brief Visitors which track the device for the current Relay expression.
 */

#include "./device_aware_visitors.h"

namespace tvm {
namespace relay {
namespace transform {

// TODO(mbs): This machinery can be used a) on expressions/modules which have not had
// device planning run, and b) on expressions for which we've not kept track of their
// containing module. For now we'll handle b) by being forgiving as possible when recovering
// the device for an expression, and we'll support a) the same way. But better would be
// to ICHECK fail when, eg, a variable is not in scope or the lexical device stack is empty.

LexicalOnDeviceMixin::LexicalOnDeviceMixin(const Optional<IRModule>& maybe_mod) {
  if (maybe_mod) {
    for (const auto& pair : maybe_mod.value()->functions) {
      if (const auto* function_node = pair.second.as<FunctionNode>()) {
        DLDeviceType device_type = GetFunctionResultDeviceType(function_node);
        if (device_type != kInvalidDeviceType) {
          global_var_device_types_.emplace(pair.first, device_type);
        }
      }
    }
  }
}

DLDeviceType LexicalOnDeviceMixin::GetInScopeDeviceType(const Expr& expr) const {
  auto props = GetOnDeviceProps(expr);
  if (props.body.defined() && props.is_fixed) {
    return props.device_type;
  } else if (const auto* var_node = expr.as<VarNode>()) {
    // Lookup variable binding.
    auto itr = var_device_types_.find(GetRef<Var>(var_node));
    if (itr != var_device_types_.end()) {
      return itr->second;
    }
    // else: fallthrough to unknown
  } else if (const auto* global_var_node = expr.as<GlobalVarNode>()) {
    // Lookup global variable.
    auto itr = global_var_device_types_.find(GetRef<GlobalVar>(global_var_node));
    if (itr != global_var_device_types_.end()) {
      return itr->second;
    }
    // else: fallthrough to unknown
  } else {
    if (!expr_device_types_.empty()) {
      // Use the currently in-scope device type.
      return expr_device_types_.back();
    }
    // else: fallthrough to unknown
  }
  return kInvalidDeviceType;
}

void LexicalOnDeviceMixin::EnterFunctionBody() { ++function_nesting_; }

void LexicalOnDeviceMixin::ExitFunctionBody() {
  ICHECK_GT(function_nesting_, 0);
  --function_nesting_;
}

void LexicalOnDeviceMixin::PushDeviceType(DLDeviceType device_type) {
  if (device_type == kInvalidDeviceType) {
    return;
  }
  expr_device_types_.emplace_back(device_type);
}

void LexicalOnDeviceMixin::PopDeviceType() {
  if (expr_device_types_.empty()) {
    return;
  }
  expr_device_types_.pop_back();
}

void LexicalOnDeviceMixin::PushBoundVar(Var var, DLDeviceType device_type) {
  if (device_type == kInvalidDeviceType) {
    return;
  }
  ICHECK(var_device_types_.find(var) == var_device_types_.end());
  var_device_types_.emplace(std::move(var), device_type);
}

void LexicalOnDeviceMixin::PopBoundVar(const Var& var) {
  auto itr = var_device_types_.find(var);
  if (itr == var_device_types_.end()) {
    return;
  }
  var_device_types_.erase(itr);
}

// TODO(mbs): We'd probably have less tedious code duplication if we redefined the memoizing
// mutator on top of the generic Functor.

void DeviceAwareExprVisitor::VisitExpr_(const FunctionNode* function_node) {
  if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
    // No tracking inside primitive functions.
    DeviceAwareVisitExpr_(function_node);
  } else {
    // Function parameters come into scope.
    for (size_t i = 0; i < function_node->params.size(); ++i) {
      PushBoundVar(function_node->params[i], GetFunctionParamDeviceType(function_node, i));
    }
    // Entering scope of function body.
    PushDeviceType(GetFunctionResultDeviceType(function_node));
    EnterFunctionBody();

    DeviceAwareVisitExpr_(function_node);

    // Leaving scope of function body.
    ExitFunctionBody();
    PopDeviceType();
    // Function parameters go out of scope.
    for (size_t i = 0; i < function_node->params.size(); ++i) {
      PopBoundVar(function_node->params[i]);
    }
  }
}

void DeviceAwareExprVisitor::VisitExpr_(const LetNode* let_node) {
  PreVisitLetBlock_(let_node);
  std::vector<const LetNode*> bindings;
  Expr expr = GetRef<Expr>(let_node);
  while (const auto* inner_let_node = expr.as<LetNode>()) {
    // Let-bound var (in pre visited version) goes into scope.
    // (We'll just assume this is a letrec).
    PushBoundVar(inner_let_node->var, GetInScopeDeviceType(inner_let_node->value));
    PreVisitLetBinding_(inner_let_node->var, inner_let_node->value);
    bindings.emplace_back(inner_let_node);
    expr = inner_let_node->body;
  }

  VisitExpr(expr);

  for (auto itr = bindings.rbegin(); itr != bindings.rend(); ++itr) {
    // Let-bound var goes out of scope.
    PopBoundVar((*itr)->var);
    PostVisitLet_(*itr);
  }
  PostVisitLetBlock_(let_node);
}

void DeviceAwareExprVisitor::VisitExpr_(const CallNode* call_node) {
  auto props = GetOnDeviceProps(call_node);
  if (props.body.defined() && props.is_fixed) {
    // Entering lexical scope of fixed "on_device" call.
    PushDeviceType(props.device_type);
    VisitExpr(props.body);
    // Leaving lexical scope of "on_device" call.
    PopDeviceType();
  } else {
    DeviceAwareVisitExpr_(call_node);
  }
}

void DeviceAwareExprVisitor::DeviceAwareVisitExpr_(const FunctionNode* function_node) {
  ExprVisitor::VisitExpr_(function_node);
}

void DeviceAwareExprVisitor::DeviceAwareVisitExpr_(const CallNode* call_node) {
  ExprVisitor::VisitExpr_(call_node);
}

void DeviceAwareExprVisitor::PreVisitLetBlock_(const LetNode* let_node) {
  // no-op
}

void DeviceAwareExprVisitor::PreVisitLetBinding_(const Var& var, const Expr& value) {
  VisitExpr(var);
  VisitExpr(value);
}

void DeviceAwareExprVisitor::PostVisitLet_(const LetNode* let_node) {
  // no-op
}

void DeviceAwareExprVisitor::PostVisitLetBlock_(const LetNode* let_node) {
  // no-op
}

Expr DeviceAwareExprMutator::VisitExpr_(const FunctionNode* function_node) {
  if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
    // No tracking inside primitive functions.
    return DeviceAwareVisitExpr_(function_node);
  } else {
    // Function parameters come into scope.
    for (size_t i = 0; i < function_node->params.size(); ++i) {
      PushBoundVar(function_node->params[i], GetFunctionParamDeviceType(function_node, i));
    }
    // Entering scope of function body.
    PushDeviceType(GetFunctionResultDeviceType(function_node));
    EnterFunctionBody();

    Expr result = DeviceAwareVisitExpr_(function_node);

    // Leaving scope of function body.
    ExitFunctionBody();
    PopDeviceType();
    // Function parameters go out of scope.
    for (size_t i = 0; i < function_node->params.size(); ++i) {
      PopBoundVar(function_node->params[i]);
    }

    return result;
  }
}

Expr DeviceAwareExprMutator::VisitExpr_(const LetNode* let_node) {
  PreVisitLetBlock_(let_node);
  std::vector<std::tuple<Var, Expr, Span, const LetNode*>> bindings;
  Expr expr = GetRef<Expr>(let_node);
  while (const auto* inner_let_node = expr.as<LetNode>()) {
    // Let-bound var (in pre visited version) goes into scope.
    // (We'll just assume this is a letrec.)
    PushBoundVar(inner_let_node->var, GetInScopeDeviceType(inner_let_node->value));
    std::pair<Var, Expr> pair = PreVisitLetBinding_(inner_let_node->var, inner_let_node->value);
    bindings.emplace_back(pair.first, pair.second, inner_let_node->span, inner_let_node);
    expr = inner_let_node->body;
  }

  expr = VisitExpr(expr);

  for (auto itr = bindings.rbegin(); itr != bindings.rend(); ++itr) {
    // Let-bound var goes out of scope.
    const LetNode* pre_let_node = std::get<3>(*itr);
    PopBoundVar(pre_let_node->var);
    Let post_let = Let(/*var=*/std::get<0>(*itr), /*value=*/std::get<1>(*itr),
                       /*body=*/expr, /*span=*/std::get<2>(*itr));
    expr = PostVisitLet_(pre_let_node, post_let.get());
  }
  return PostVisitLetBlock_(let_node, expr.as<LetNode>());
}

Expr DeviceAwareExprMutator::VisitExpr_(const CallNode* call_node) {
  auto props = GetOnDeviceProps(call_node);
  if (props.body.defined() && props.is_fixed) {
    // Entering lexical scope of fixed "on_device" call.
    PushDeviceType(props.device_type);
    Expr expr = VisitExpr(props.body);
    // Leaving lexical scope of "on_device" call.
    PopDeviceType();
    return MaybeOnDevice(expr, props.device_type, props.is_fixed);
  } else {
    return DeviceAwareVisitExpr_(call_node);
  }
}

Expr DeviceAwareExprMutator::DeviceAwareVisitExpr_(const FunctionNode* function_node) {
  return ExprMutator::VisitExpr_(function_node);
}

Expr DeviceAwareExprMutator::DeviceAwareVisitExpr_(const CallNode* call_node) {
  return ExprMutator::VisitExpr_(call_node);
}

void DeviceAwareExprMutator::PreVisitLetBlock_(const LetNode* let_node) { /* no-op */
}

std::pair<Var, Expr> DeviceAwareExprMutator::PreVisitLetBinding_(const Var& var,
                                                                 const Expr& value) {
  return std::make_pair(Downcast<Var>(VisitExpr(var)), VisitExpr(value));
}

Expr DeviceAwareExprMutator::PostVisitLet_(const LetNode* pre_let_node,
                                           const LetNode* post_let_node) {
  if (pre_let_node->var == post_let_node->var && pre_let_node->value == post_let_node->value &&
      pre_let_node->body == post_let_node->body) {
    return GetRef<Expr>(pre_let_node);
  } else {
    return GetRef<Expr>(post_let_node);
  }
}

Expr DeviceAwareExprMutator::PostVisitLetBlock_(const LetNode* pre_let_node,
                                                const LetNode* post_let_node) {
  if (pre_let_node->var == post_let_node->var && pre_let_node->value == post_let_node->value &&
      pre_let_node->body == post_let_node->body) {
    return GetRef<Expr>(pre_let_node);
  } else {
    return GetRef<Expr>(post_let_node);
  }
}

}  // namespace transform
}  // namespace relay
}  // namespace tvm
