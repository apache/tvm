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
 * \file src/relay/transforms/device_aware_visitors.h
 * \brief Visitors which track the device for the current Relay expression and Relay Vars.
 */

#ifndef TVM_RELAY_TRANSFORMS_DEVICE_AWARE_VISITORS_H_
#define TVM_RELAY_TRANSFORMS_DEVICE_AWARE_VISITORS_H_

#include <dlpack/dlpack.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/function.h>

#include <unordered_map>
#include <utility>
#include <vector>

#include "../op/annotation/annotation.h"

namespace tvm {
namespace relay {
namespace transform {

/*!
 * \brief Helper class for expression transformers which need to keep track of the device
 * holding the results of expressions and bound variables. This is recovered from the
 * "on_device" function attributes and fixed "on_device" CallNodes added by the PlanDevices
 * pass.
 *
 * \sa \p DeviceAwareExpr{Visitor,Mutator}.
 */
class LexicalOnDeviceMixin {
 protected:
  /*!
   * \brief Returns the device type on which the result of \p expr should/will be stored, assuming
   * Push/Pop DeviceType/BoundVar have been correctly called. Returns \p kInvalidDeviceType if
   * stack is empty and no bound vars have device types.
   */
  DLDeviceType GetInScopeDeviceType(const Expr& expr) const;

  /*! \brief Indicate a function body is being entered. */
  void EnterFunctionBody();

  /*! \brief Indicate a function body has been processed. */
  void ExitFunctionBody();

  /*! \brief Push a device type onto the lexical device stack. Ignore if \p kInvalidDeviceType. */
  void PushDeviceType(const DLDeviceType device_type);

  /*! \brief Pop a device type from the lexical device stack. Ignore if stack is empty. */
  void PopDeviceType();

  /*! \brief Remember that \p var will be stored on \p device_type. Ignore if \p kInvalidDeviceType.
   *
   * CAUTION: Despite the name we don't support re-entering the same function body.
   */
  void PushBoundVar(Var var, DLDeviceType device_type);

  /*! \brief Remove the binding for \p var to it's device type. Ignore if var is not bound. */
  void PopBoundVar(const Var& var);

  /*!
   * \brief Returns the number of function definitions wrapping the currently visited expression.
   */
  int function_nesting() const { return function_nesting_; }

 private:
  /*!
   * \brief The number of function bodies entered. Since many transforms need to distinguish global
   * functions from local functions this supports the mixin's \p is_global() helper method.
   */
  int function_nesting_ = 0;

  /*!
   * \brief The stack of lexically enclosing "on_device" devices types, from outermost to innermost.
   * When visiting an expression other than a variable we can assume the expression result is
   * to be stored on device_type_.back().
   */
  std::vector<DLDeviceType> expr_device_types_;
  /*!
   * \brief A map from in-scope variable to their device types. We may assume the variable is only
   * ever bound to a value stored on this device at runtime.
   */
  std::unordered_map<Var, DLDeviceType, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>
      var_device_types_;
};

template <typename FType>
class DeviceAwareExprFunctor;

/*!
 * \brief ExprFunctor which tracks devices. We only support 'visitor' style implementation
 * with no additional arguments, thus this is equivalent to \p DeviceAwareExprVisitor without
 * any memoization.
 */
template <>
class DeviceAwareExprFunctor<void(const Expr& n)> : public ExprFunctor<void(const Expr& n)>,
                                                    public LexicalOnDeviceMixin {
 private:
  using TSuper = ExprFunctor<void(const Expr& n)>;

 public:
  void VisitExpr_(const FunctionNode* function_node) {
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

  void VisitExpr_(const LetNode* let_node) {
    PreVisitLetBlock_(let_node);
    std::vector<const LetNode*> bindings;
    Expr expr = GetRef<Expr>(let_node);
    while (const auto* inner_let_node = expr.as<LetNode>()) {
      // Let-bound var (in pre visited version) goes into scope.
      // (We'll just assume this is a letrec.)
      PushBoundVar(inner_let_node->var, GetInScopeDeviceType(inner_let_node->value));
      PreVisitLetBinding_(inner_let_node->var, inner_let_node->value);
      bindings.emplace_back(inner_let_node);
      expr = inner_let_node->body;
    }

    VisitExpr(expr);

    for (auto itr = bindings.rbegin(); itr != bindings.rend(); ++itr) {
      // Let-bound var goes out of scope.
      const LetNode* visited_let_node = *itr;
      PopBoundVar(visited_let_node->var);
      PostVisitLet_(visited_let_node);
    }
    PostVisitLetBlock_(let_node);
  }

  void VisitExpr_(const CallNode* call_node) {
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

  /*!
   * \brief These are as for VisitExpr_. Devices for expressions and function parameters will be
   * tracked automatically. Default implementation defers to ExprMutator::VisitExpr_. For
   * functions the function_nesting count will already include that of \p function_node.
   */

  virtual void DeviceAwareVisitExpr_(const FunctionNode* function_node) {
    return TSuper::VisitExpr_(function_node);
  }

  virtual void DeviceAwareVisitExpr_(const CallNode* call_node) {
    return TSuper::VisitExpr_(call_node);
  }

  /*!
   * \brief Visit the first let in a chain of let expressions before any let bindings or final
   * body has been visited. Default implementation is a no-op.
   */
  virtual void PreVisitLetBlock_(const LetNode* let_node) { /* no-op */
  }

  /*!
   * \brief Visit a let-bound expression before the let body has been visited. Devices for the
   * let-bound variable will be tracked automatically. Default implementation just visits var and
   * value.
   */
  virtual void PreVisitLetBinding_(const Var& var, const Expr& value) {
    VisitExpr(var);
    VisitExpr(value);
  }

  /*!
   * \brief Visit a let expression after the let-bound value and body have been visited.
   * Default implementation is a no-op.
   */
  virtual void PostVisitLet_(const LetNode* let_node) { /* no-op */
  }

  /*!
   * \brief Visit the first let in a chain of let expressions after it has been visited.
   * Default implementation is a no-op.
   */
  virtual void PostVisitLetBlock_(const LetNode* let_node) {}
};

/*! \brief ExprVisitor which tracks devices. */
class DeviceAwareExprVisitor : public ExprVisitor, public LexicalOnDeviceMixin {
 public:
  using ExprVisitor::VisitExpr_;

  void VisitExpr_(const FunctionNode* function_node) final;
  void VisitExpr_(const LetNode* let_node) final;
  void VisitExpr_(const CallNode* call_node) final;

  /*!
   * \brief These are as for VisitExpr_. Devices for expressions and function parameters will be
   * tracked automatically. Default implementation defers to ExprMutator::VisitExpr_. For
   * functions the function_nesting count will already include that of \p function_node.
   */
  virtual void DeviceAwareVisitExpr_(const FunctionNode* function_node);
  virtual void DeviceAwareVisitExpr_(const CallNode* call_node);

  /*!
   * \brief Visit the first let in a chain of let expressions before any let bindings or final
   * body has been visited. Default implementation is a no-op.
   */
  virtual void PreVisitLetBlock_(const LetNode* let_node);

  /*!
   * \brief Visit a let-bound expression before the let body has been visited. Devices for the
   * let-bound variable will be tracked automatically. Default implementation just visits var and
   * value.
   */
  virtual void PreVisitLetBinding_(const Var& var, const Expr& value);

  /*!
   * \brief Visit a let expression after the let-bound value and body have been visited.
   * Default implementation is a no-op.
   */
  virtual void PostVisitLet_(const LetNode* let_node);

  /*!
   * \brief Visit the first let in a chain of let expressions after it has been visited.
   * Default implementation is a no-op.
   */
  virtual void PostVisitLetBlock_(const LetNode* let_node);
};

/*! \brief ExprMutator which tracks devices. */
class DeviceAwareExprMutator : public ExprMutator, public LexicalOnDeviceMixin {
 public:
  Expr VisitExpr_(const FunctionNode* function_node) final;
  Expr VisitExpr_(const LetNode* let_node) final;
  Expr VisitExpr_(const CallNode* call_node) final;

  /*!
   * \brief These are as for VisitExpr_. Devices for expressions and function parameters will be
   * tracked automatically. Default implementation defers to ExprMutator::VisitExpr_. For
   * functions the function_nesting count will already include that of \p function_node.
   */
  virtual Expr DeviceAwareVisitExpr_(const FunctionNode* function_node);
  virtual Expr DeviceAwareVisitExpr_(const CallNode* call_node);

  /*!
   * \brief Visit the first let in a chain of let expressions before any let bindings or final
   * body has been visited. Default implementation is a no-op.
   */
  virtual void PreVisitLetBlock_(const LetNode* let_node);

  /*!
   * \brief Visit a let-bound expression before the let body has been visited. Devices for the
   * let-bound variable will be tracked automatically. Default implementation just visits var and
   * value.
   */
  virtual std::pair<Var, Expr> PreVisitLetBinding_(const Var& var, const Expr& value);

  /*!
   * \brief Visit a let expression after the let-bound value and body have been visited.
   * Default implementation just returns a reference to the post-visited node.
   */
  virtual Expr PostVisitLet_(const LetNode* pre_let_node, const LetNode* post_let_node);

  /*!
   * \brief Visit the first let in a chain of let expressions after it has been visited.
   * Default implementation returns reference to let node.
   */
  virtual Expr PostVisitLetBlock_(const LetNode* pre_let_node, const LetNode* post_let_node);
};

}  // namespace transform
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TRANSFORMS_DEVICE_AWARE_VISITORS_H_
