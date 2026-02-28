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
 * \file tvm/support/with.h
 * \brief RAII wrapper function to enter and exit a context object
 *        similar to python's with syntax.
 */
#ifndef TVM_SUPPORT_WITH_H_
#define TVM_SUPPORT_WITH_H_

#include <exception>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace tvm {

/*!
 * \brief RAII wrapper function to enter and exit a context object
 *        similar to python's with syntax.
 *
 * \code
 * // context class
 * class MyContext {
 *  private:
 *    friend class With<MyContext>;
      MyContext(arguments);
 *    void EnterWithScope();
 *    void ExitWithScope();
 * };
 *
 * {
 *   With<MyContext> scope(arguments);
 *   // effect take place.
 * }
 * \endcode
 *
 * \tparam ContextType Type of the context object.
 */
template <typename ContextType>
class With {
 public:
  /*!
   * \brief constructor.
   *  Enter the scope of the context.
   */
  template <typename... Args>
  explicit With(Args&&... args) : ctx_(std::forward<Args>(args)...) {
    ctx_.EnterWithScope();
  }
  /*! \brief destructor, leaves the scope of the context. */
  ~With() noexcept(false) { ctx_.ExitWithScope(); }

  // Disable copy and move construction.  `With` is intended only for
  // use in nested contexts that are exited in the reverse order of
  // entry.  Allowing context to be copied or moved would break this
  // expectation.
  With(const With& other) = delete;
  With& operator=(const With& other) = delete;
  With(With&& other) = delete;
  With& operator=(With&& other) = delete;

  ContextType* get() { return &ctx_; }
  const ContextType* get() const { return &ctx_; }

  ContextType* operator->() { return get(); }
  const ContextType* operator->() const { return get(); }
  ContextType& operator*() { return *get(); }
  const ContextType* operator*() const { return *get(); }

  ContextType operator()() { return ctx_; }

 private:
  /*! \brief internal context type. */
  ContextType ctx_;
};

/*!
 * \brief A group of RAII contexts managed together.
 *
 * Allows dynamically emplacing multiple context objects that are
 * all exited (in reverse order) when the group is destroyed.
 * ContextType must declare `friend class With<ContextType>`
 * and provide EnterWithScope() / ExitWithScope() methods.
 *
 * \code
 *   WithGroup<ConstraintContext> group;
 *   group.Emplace(&analyzer, cond1);  // constructs and enters
 *   group.Emplace(&analyzer, cond2);  // constructs and enters
 *   // destructor: exits cond2, then cond1
 * \endcode
 *
 * \tparam ContextType The context type with EnterWithScope/ExitWithScope.
 */
template <typename ContextType>
class WithGroup {
 public:
  WithGroup() = default;
  WithGroup(WithGroup&&) = default;
  WithGroup& operator=(WithGroup&&) = default;
  WithGroup(const WithGroup&) = delete;
  WithGroup& operator=(const WithGroup&) = delete;

  /*!
   * \brief Construct a context and enter its scope.
   * \param args Arguments forwarded to ContextType constructor.
   */
  template <typename... Args>
  void Emplace(Args&&... args) {
    entries_.push_back(std::make_unique<With<ContextType>>(std::forward<Args>(args)...));
  }

  /*! \brief Number of active contexts in this group. */
  size_t size() const { return entries_.size(); }

  /*!
   * \brief Destructor — exits all contexts in reverse order.
   *
   * On normal exit: if any ExitWithScope throws, the remaining
   * contexts are still cleaned up, then the first exception
   * is re-thrown.
   *
   * During stack unwinding: all exceptions are swallowed
   * to avoid std::terminate.
   */
  ~WithGroup() noexcept(false) {
    bool unwinding = std::uncaught_exceptions() > 0;
    std::exception_ptr first_exc;
    while (!entries_.empty()) {
      // Move the last entry out of the vector first, then destroy it.
      // This ensures entries_ shrinks even if ~With() throws.
      auto entry = std::move(entries_.back());
      entries_.pop_back();
      try {
        entry.reset();  // calls ~With<ContextType>() -> ExitWithScope()
      } catch (...) {
        if (!unwinding && !first_exc) {
          first_exc = std::current_exception();
        }
      }
    }
    if (first_exc) std::rethrow_exception(first_exc);
  }

 private:
  std::vector<std::unique_ptr<With<ContextType>>> entries_;
};

}  // namespace tvm
#endif  // TVM_SUPPORT_WITH_H_
