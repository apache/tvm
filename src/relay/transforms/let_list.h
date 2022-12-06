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
 * \file let_list.h
 * \brief LetList record let binding and insert let expression implicitly.
 *  using it, one can treat AST as value instead of expression,
 *  and pass them around freely without fear of AST explosion (or effect duplication).
 *  for example, if one write 'b = a + a; c = b + b; d = c + c', the AST will contain 8 'a'.
 *  if one instead write 'b = ll.Push(a + a); c = ll.Push(b + b); d = ll.Get(c + c);',
 *  the AST will contain 2 'a', as b and c are now variables.
 */
#ifndef TVM_RELAY_TRANSFORMS_LET_LIST_H_
#define TVM_RELAY_TRANSFORMS_LET_LIST_H_

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tvm/relay/type.h"

namespace tvm {
namespace relay {

/*!
 * \brief LetList allow you to transform expression into variables, so you can copy them around.
 *  one can insert into the LetList by calling Push, and wrap an expression with bindings with Get.
 *  additionally, there is the 'With' function, which automatically call Get.
 */
class LetList {
 public:
  ~LetList() {
    if (lets_.size() > 0 && !used_) {
      LOG(WARNING) << "letlist not used";
    }
  }
  /*!
   * \brief insert a binding.
   *
   * \param pv the var of the binding.
   *
   * \param expr the value of the binding.
   *
   * \return a Var that hold the inserted expr.
   */
  Var Push(Var pv, Expr expr) {
    ICHECK(!used_);
    ICHECK(WellFormed(expr)) << "expression:" << std::endl << PrettyPrint(expr);
    lets_.emplace_back(std::make_pair(pv, expr));
    return pv;
  }

  /*!
   * \brief insert a binding.
   *
   * \param expr the value of the binding.
   *
   * \param ty the type of the binding.
   *
   * \return a Var that hold the inserted expr.
   */
  Var Push(Expr expr, Type ty) { return Push(Var::GenSym(ty), expr); }

  /*!
   * \brief insert a binding.
   *
   *  \param expr the value of the binding.
   *
   *  \return a Var that hold the inserted expr.
   */
  Var Push(Expr expr) { return Push(expr, Type()); }

  /*!
   * \brief wrap an expr around the LetList.
   *
   *  \param body the Expression to be wrapped around.
   *
   *  \return the wrapped expr.
   */
  Expr Get(const Expr& body) {
    ICHECK(!used_);
    Expr ret = body;
    for (auto rit = lets_.rbegin(); rit != lets_.rend(); ++rit) {
      ret = Let(std::get<0>(*rit), std::get<1>(*rit), ret);
    }
    used_ = true;
    return ret;
  }

  /*! \brief get the number of let bindings in the let list.
   *
   *  \return the let list size.
   */
  size_t size() const { return lets_.size(); }

  /*! \brief generate an LetList and wrap the result automatically.
   *
   *  \param f a function that generate the unwrapped Expr.
   *
   *  \code
   *  // Example code that generate `16 * a` using 4 plus instead of 15 plus.
   *  Expr mult_sixteen(const Var& a) {
   *    Op plus = Op::Get("plus");
   *    // Automatically call Get with LetList::With
   *    return LetList::With([&](LetList* ll) {
   *      // Turn a call to plus into a variable to avoid duplication of code
   *      Var b = ll->Push(Call(plus, {a, a}));
   *      Var c = ll->Push(Call(plus, {b, b}));
   *      Var d = ll->Push(Callplus, {c, c}));
   *      return Call(plus, {d, d});
   *    });
   *  }
   *  \endcode
   *
   *  \return the wrapped Expr.
   */
  template <typename F>
  static Expr With(F&& f) {
    LetList ll;
    return ll.Get(f(&ll));
  }

  static Expr LetBind(const Expr& e, const std::function<Expr(const Var&)>& f) {
    return With([&](LetList* ll) { return f(ll->Push(e)); });
  }

 private:
  std::vector<std::pair<Var, Expr>> lets_;
  bool used_ = false;
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_TRANSFORMS_LET_LIST_H_
