/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_pass.h
 * \brief Collection of IR pass functions and visit functions
 */
#ifndef TVM_IR_PASS_H_
#define TVM_IR_PASS_H_

#include <tvm/ir_node.h>
#include <unordered_map>
#include "./expr.h"

namespace tvm {
namespace ir {

/*!
 * \brief Substitute occurance of IRNode in expr
 * \param replacements The replacement rule of substitution
 * \param expr The expression to be substituted.
 */
Expr Substitute(const std::unordered_map<const IRNode*, Expr>& replacements, Expr expr);

}  // namespace ir
}  // namespace tvm

#endif  // TVM_IR_PASS_H_
