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
 * \brief verifies whether the IR stmt or Expr is in SSA form.
 *  That is: each VarExpr is defined and assigned once(in Let/For)
 *
 * \param ir The root of the IR DAG.
 * \return Whether IR is in SSA form.
 * \note All the passes in this file uses SSA form and outputs SSA form.
 */
bool VerifySSA(const IRNodeRef& ir);

/*!
 * \brief Convert a IR node to be SSA form.
 * \param stmt The source statement to be converted.
 * \return The converted form.
 */
Stmt ConvertSSA(const Stmt& stmt);

/*!
 * \brief inline all calls of f in stmt.
 *
 * \param f The function reference to be inlined
 * \param args The arguments variable of the function.
 * \param body The defintion body of the function.
 * \param stmt The statement to apply inline optimization.
 * \return The result stmt
 *
 * \note All the passes in this file uses SSA form and outputs SSA form.
 */
Stmt InlineSSA(FunctionRef f, const std::vector<Var>& args, Expr body, Stmt stmt);


}  // namespace ir
}  // namespace tvm

#endif  // TVM_IR_PASS_H_
