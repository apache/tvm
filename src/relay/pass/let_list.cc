/*!
 *  Copyright (c) 2018 by Contributors
 * \file let_list.cc
 * \brief this file is the example in the `With` function of letlist.
 *  it is only to makesure everything compile -
 *  this is temporary, as letlist is used heavily across M.K.'s other pr,
 *  once we merge them, we have test of letlist, and can safely remove this file.
 */

#include "./let_list.h"
#include <tvm/relay/op.h>

namespace tvm {
namespace relay {

// Example code that generate `16 * a` using 4 plus instead of 15 plus.
Expr mult_sixteen(const Var& a) {
  Op plus = Op::Get("plus");
  // Automatically call Get with LetList::With
  return LetList::With([&](LetList* ll) {
    // Turn a call to plus into a variable to avoid duplication of code
    Var b = ll->Push(CallNode::make(plus, {a, a}));
    Var c = ll->Push(CallNode::make(plus, {b, b}));
    Var d = ll->Push(CallNode::make(plus, {c, c}));
    return CallNode::make(plus, {d, d});
  });
}

}  // namespace relay
}  // namespace tvm
