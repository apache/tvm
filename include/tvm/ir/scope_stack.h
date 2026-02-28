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
 * \file tvm/ir/scope_stack.h
 * \brief A generic scope stack for managing hierarchical state during IR visiting.
 */
#ifndef TVM_IR_SCOPE_STACK_H_
#define TVM_IR_SCOPE_STACK_H_

#include <tvm/ffi/error.h>

#include <deque>
#include <type_traits>

namespace tvm {

/*!
 * \brief A scope stack for maintaining hierarchical state during IR visiting.
 *
 * During IR tree traversal, visitors often need to track scope-local state
 * (e.g., active constraints, variable bindings) that should be automatically
 * cleaned up when leaving a scope. ScopeStack provides this via WithNewScope,
 * which pushes a new element on entry and pops it on exit.
 *
 * \code
 *   ScopeStack<WithGroup<ConstraintContext>> constraints;
 *
 *   // In VisitStmt_(ForNode):
 *   return constraints.WithNewScope([&]() -> Stmt {
 *     constraints.Current().Emplace(&analyzer, condition);
 *     return StmtExprMutator::VisitStmt_(op);
 *   });
 * \endcode
 *
 * \tparam T The element type stored on the stack. Must be default-constructible.
 */
template <typename T>
class ScopeStack {
 public:
  /*! \brief Construct with one initial scope level. */
  ScopeStack() { stack_.emplace_back(); }

  /*! \brief Return the number of active scopes. */
  size_t size() const { return stack_.size(); }

  /*! \brief Return true if no scopes are active. */
  bool empty() const { return stack_.empty(); }

  /*!
   * \brief Access the current (innermost) scope element.
   *
   * The returned reference is stable across push_back/pop_back because
   * std::deque guarantees pointer stability for these operations.
   *
   * \return Mutable reference to the top element.
   */
  T& Current() {
    TVM_FFI_ICHECK(!stack_.empty());
    return stack_.back();
  }

  /*! \brief Const access to the current (innermost) scope element. */
  const T& Current() const {
    TVM_FFI_ICHECK(!stack_.empty());
    return stack_.back();
  }

  /*!
   * \brief Execute body within a new scope.
   *
   * Pushes a new T onto the stack, executes the body, then pops it.
   *
   * \param body A callable to execute within the scope.
   * \return The return value of body(), if non-void.
   */
  template <typename F>
  auto WithNewScope(F&& body) -> decltype(body()) {
    stack_.emplace_back();
    struct Guard {
      std::deque<T>* stack;
      ~Guard() noexcept(false) { stack->pop_back(); }
    } guard{&stack_};
    if constexpr (std::is_void_v<decltype(body())>) {
      body();
    } else {
      return body();
    }
  }

 private:
  /*!
   * \brief The scope stack.
   *
   * We use std::deque rather than std::vector for pointer stability:
   * references returned by Current() remain valid across push/pop operations.
   * This is critical because methods called on Current() (e.g., Emplace on
   * a WithGroup) may trigger re-entrant code that pushes new scopes onto
   * the same stack. With std::vector the internal buffer reallocation would
   * invalidate the reference, causing use-after-free.
   */
  std::deque<T> stack_;
};

}  // namespace tvm

#endif  // TVM_IR_SCOPE_STACK_H_
