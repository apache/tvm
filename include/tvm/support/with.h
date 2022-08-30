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

#include <dmlc/common.h>

#include <functional>
#include <utility>

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
  ~With() DMLC_THROW_EXCEPTION { ctx_.ExitWithScope(); }

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
 * \brief A context type that delegates EnterWithScope and ExitWithScope
 *        to user-provided functions.
 */
class ContextManager {
 public:
  /*!
   * \brief Constructor of ContextManager.
   * \param f_enter The function to call when entering scope. If it's nullptr, do nothing when
   *                entering.
   * \param f_exit The function to call when exiting scope. If it's nullptr, do nothing
   *               when exiting.
   */
  template <class FEnter, class FExit>
  explicit ContextManager(FEnter f_enter, FExit f_exit) : f_enter_(f_enter), f_exit_(f_exit) {}

 private:
  void EnterWithScope() {
    if (f_enter_) f_enter_();
  }
  void ExitWithScope() {
    if (f_exit_) f_exit_();
  }
  std::function<void()> f_enter_;
  std::function<void()> f_exit_;
  template <typename>
  friend class With;
};

}  // namespace tvm
#endif  // TVM_SUPPORT_WITH_H_
