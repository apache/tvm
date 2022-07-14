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
#include "../op/memory/on_device.h"

namespace tvm {
namespace relay {
namespace transform {

/*!
 * \brief Helper class for expression transformers which need to keep track of the \p VirtualDevice
 * holding the results of expressions. This is recovered from function attributes and "on_device"
 * CallNodes added by the PlanDevices pass.
 *
 * \sa \p DeviceAwareExpr{Functor,Visitor,Mutator}.
 */
class LexicalOnDeviceMixin {
 protected:
  explicit LexicalOnDeviceMixin(const Optional<IRModule>& maybe_mod);

  /*!
   * \brief Returns the \p VirtualDevice on which the result of \p expr should/will be stored,
   * assuming {Push,Pop}{VirtualDevice,BoundVar} have been correctly called. May return the
   * unconstrained \p VirtualDevice if the device planning pass has not been run.
   */
  VirtualDevice GetVirtualDevice(const Expr& expr) const;

  /*! \brief Indicate a function body is being entered. */
  void EnterFunctionBody();

  /*! \brief Indicate a function body has been processed. */
  void ExitFunctionBody();

  /*! \brief Push an \p VirtualDevice onto the lexical VirtualDevice stack. Ignore if unconstrained.
   */
  void PushVirtualDevice(const VirtualDevice& virtual_device);

  /*! \brief Pop an \p VirtualDevice from the lexical VirtualDevice stack. Ignore if stack is empty.
   */
  void PopVirtualDevice();

  /*! \brief Remember that \p var will be stored at \p virtual_device. Ignore if unconstrained.
   *
   * CAUTION: Despite the name we don't support re-entering the same function body.
   */
  void PushBoundVar(Var var, const VirtualDevice& virtual_device);

  /*! \brief Remove the binding for \p var to its \p VirtualDevice. Ignore if var is not bound. */
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
   * \brief The stack of lexically enclosing "on_device" \p VirtualDevices, from outermost to
   * innermost. When visiting an expression other than a variable we can assume the expression's
   * result is to be stored on \p expr_virtual_devices.back().
   */
  std::vector<VirtualDevice> expr_virtual_devices_;

  /*!
   * \brief A map from in-scope local variables to their \p VirtualDevices. We may assume the
   * variable is only ever bound to a value stored on this \p VirtualDevice at runtime.
   *
   * Note: We're playing it safe and keying by object refs here just in case the Relay expression
   * being rewritten has no module or other global to keep it alive.
   */
  std::unordered_map<Var, VirtualDevice, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>
      var_virtual_devices_;

  /*!
   * \brief A map from global variables to their \p VirtualDevices, ie the "result_virtual_device"
   * of the function they are bound to in the module we are working on. We calculate and store this
   * explicitly so that we don't need to hold on to any module, which is often in the process of
   * being rewritten.
   */
  std::unordered_map<GlobalVar, VirtualDevice, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>
      global_var_virtual_devices_;
};

template <typename FType>
class DeviceAwareExprFunctor;

/*!
 * \brief ExprFunctor which tracks \p VirtualDevices. We only support 'visitor' style implementation
 * with no additional arguments, thus this is equivalent to \p DeviceAwareExprVisitor without
 * any memoization.
 */
template <>
class DeviceAwareExprFunctor<void(const Expr& n)> : public ExprFunctor<void(const Expr& n)>,
                                                    public LexicalOnDeviceMixin {
 private:
  using TSuper = ExprFunctor<void(const Expr& n)>;

 public:
  explicit DeviceAwareExprFunctor(const Optional<IRModule>& maybe_mod)
      : LexicalOnDeviceMixin(maybe_mod) {}

  void VisitExpr_(const FunctionNode* function_node) {
    if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
      // No tracking inside primitive functions.
      return DeviceAwareVisitExpr_(function_node);
    } else {
      // Function parameters come into scope.
      for (auto param : function_node->params) {
        PushBoundVar(param, param->virtual_device());
      }
      // Entering scope of function body.
      VirtualDevice virtual_device = function_node->virtual_device();
      VLOG(2) << "entering " << virtual_device << " for function:" << std::endl
              << PrettyPrint(GetRef<Function>(function_node));
      PushVirtualDevice(virtual_device);
      EnterFunctionBody();

      DeviceAwareVisitExpr_(function_node);

      // Leaving scope of function body.
      ExitFunctionBody();
      PopVirtualDevice();
      VLOG(2) << "leaving " << virtual_device << " for function:" << std::endl
              << PrettyPrint(GetRef<Function>(function_node));
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
      VirtualDevice virtual_device = GetVirtualDevice(inner_let_node->value);
      VLOG(2) << "var '" << inner_let_node->var->name_hint() << "' has virtual device "
              << virtual_device;
      PushBoundVar(inner_let_node->var, virtual_device);
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
    OnDeviceProps props = GetOnDeviceProps(call_node);
    if (props.body.defined() && props.is_fixed()) {
      // Entering lexical scope of "on_device" call.
      VLOG(2) << "entering " << props.virtual_device << " for on_device:" << std::endl
              << PrettyPrint(GetRef<Call>(call_node));
      PushVirtualDevice(props.virtual_device);
      VisitExpr(props.body);
      // Leaving lexical scope of "on_device" call.
      PopVirtualDevice();
      VLOG(2) << "leaving " << props.virtual_device << " for on_device:" << std::endl
              << PrettyPrint(GetRef<Call>(call_node));
    } else {
      DeviceAwareVisitExpr_(call_node);
    }
  }

  /*!
   * \brief These are as for VisitExpr_. \p VirtualDevices for expressions and function parameters
   * will be tracked automatically. Default implementation defers to ExprMutator::VisitExpr_. For
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

/*! \brief ExprVisitor which tracks \p VirtualDevices. */
class DeviceAwareExprVisitor : public ExprVisitor, public LexicalOnDeviceMixin {
 public:
  explicit DeviceAwareExprVisitor(const Optional<IRModule>& maybe_mod)
      : LexicalOnDeviceMixin(maybe_mod) {}

  using ExprVisitor::VisitExpr_;

  void VisitExpr_(const FunctionNode* function_node) final;
  void VisitExpr_(const LetNode* let_node) final;
  void VisitExpr_(const CallNode* call_node) final;

  /*!
   * \brief These are as for VisitExpr_. \p VirtualDevices for expressions and function parameters
   * will be tracked automatically. Default implementation defers to ExprMutator::VisitExpr_. For
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
   * \brief Visit a let-bound expression before the let body has been visited. \p VirtualDevices for
   * the let-bound variable will be tracked automatically. Default implementation just visits var
   * and value.
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

/*! \brief ExprMutator which tracks \p VirtualDevices. */
class DeviceAwareExprMutator : public ExprMutator, public LexicalOnDeviceMixin {
 public:
  explicit DeviceAwareExprMutator(const Optional<IRModule>& maybe_mod)
      : LexicalOnDeviceMixin(maybe_mod) {}

  Expr VisitExpr_(const FunctionNode* function_node) final;
  Expr VisitExpr_(const LetNode* let_node) final;
  Expr VisitExpr_(const CallNode* call_node) final;

  /*!
   * \brief These are as for VisitExpr_. \p VirtualDevices for expressions and function parameters
   * will be tracked automatically. Default implementation defers to ExprMutator::VisitExpr_. For
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
   * \brief Visit a let-bound expression before the let body has been visited. \p VirtualDevices for
   * the let-bound variable will be tracked automatically. Default implementation just visits var
   * and value.
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

/*!
 * \brief Returs a map from Relay expression node to its virtual device using the annotations
 * and \p virtual_device fields of \p expr. The map's lifetime must not exceed that of
 * \p expr itself.
 */
std::unordered_map<const ExprNode*, VirtualDevice> RecoverVirtualDeviceMap(const IRModule& mod,
                                                                           const Expr& expr);

}  // namespace transform
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TRANSFORMS_DEVICE_AWARE_VISITORS_H_
