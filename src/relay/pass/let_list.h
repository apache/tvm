/*!
 *  Copyright (c) 2018 by Contributors
 * \file let_list.h
 * \brief LetList record let binding and insert let expression implicitly.
 *  using it, one can treat AST as value instead of expression,
 *  and pass them around freely without fear of AST explosion (or effect duplication).
 *  for example, if one write 'b = a + a; c = b + b; d = c + c', the AST will contain 8 'a'.
 *  if one instead write 'b = ll.Push(a + a); c = ll.Push(b + b); d = ll.Get(c + c);',
 *  the AST will contain 2 'a', as b and c are now variables.
 */
#ifndef TVM_RELAY_PASS_LET_LIST_H_
#define TVM_RELAY_PASS_LET_LIST_H_

#include <tvm/relay/expr.h>
#include <utility>
#include <vector>
#include <tuple>
#include "tvm/relay/type.h"

namespace tvm {
namespace relay {

/*! \brief LetList allow you to transform expression into variables, so you can copy them around.
 *  one can insert into the LetList by calling Push, and wrap an expression with bindings with Get.
 *  additionally, there is the 'With' function, which automatically call Get.
 */
class LetList {
 public:
  /*! \brief insert a binding.
   *
   *  \param pv the var of the binding.
   *
   *  \param ty the type of the binding.
   *
   *  \param expr the value of the binding.
   *
   *  \return a Var that hold the inserted expr.
   */
  Var Push(const Var& pv, const Type& ty, const Expr& expr) {
    std::tuple<Var, Type, Expr> tuple(pv, ty, expr);
    lets_.push_back(tuple);
    return pv;
  }

  /*! \brief insert a binding.
   *
   *  \param ty the type of the binding.
   *
   *  \param expr the value of the binding.
   *
   *  \return a Var that hold the inserted expr.
   */
  Var Push(const Type& ty, const Expr& expr) {
    return Push(VarNode::make("x"), ty, expr);
  }

  /*! \brief insert a binding.
   *
   *  \param pv the var of the binding.
   *
   *  \param expr the value of the binding.
   *
   *  \return a Var that hold the inserted expr.
   */
  Var Push(const Var& pv, const Expr& expr) {
    return Push(pv, IncompleteTypeNode::make(TypeParamNode::kType), expr);
  }

  /*! \brief insert a binding.
   *
   *  \param expr the value of the binding.
   *
   *  \return a Var that hold the inserted expr.
   */
  Var Push(const Expr& expr) {
    return Push(IncompleteTypeNode::make(TypeParamNode::kType), expr);
  }

  /*! \brief wrap an expr around the LetList.
   *
   *  \param body the Expression to be wrapped around.
   *
   *  \return the wrapped expr.
   */
  Expr Get(const Expr& body) const {
    Expr ret = body;
    for (auto rit = lets_.rbegin(); rit != lets_.rend(); ++rit) {
      ret = LetNode::make(std::get<0>(*rit), std::get<2>(*rit), ret, std::get<1>(*rit));
    }
    return ret;
  }

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
   *      Var b = ll->Push(CallNode::make(plus, {a, a}));
   *      Var c = ll->Push(CallNode::make(plus, {b, b}));
   *      Var d = ll->Push(CallNode::make(plus, {c, c}));
   *      return CallNode::make(plus, {d, d});
   *    });
   *  }
   *  \endcode
   *
   *  \return the wrapped Expr.
   */
  template<typename F>
  static Expr With(F&& f) {
    LetList ll;
    return ll.Get(f(&ll));
  }

 private:
  std::vector<std::tuple<Var, Type, Expr> > lets_;
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_LET_LIST_H_
