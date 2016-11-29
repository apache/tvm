/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_pass.h
 * \brief Collection of IR pass functions
 *
 *  All the pass functions in this file are for Stmt,
 *  We can use PassFunction(Evaluate(expr)) to apply it to Expr
 */
#ifndef TVM_IR_PASS_H_
#define TVM_IR_PASS_H_

#include <tvm/ir_functor.h>
#include <unordered_map>
#include <vector>
#include "./expr.h"
#include "./schedule.h"

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
bool VerifySSA(const Stmt& ir);

/*!
 * \brief Convert a IR node to be SSA form.
 * \param stmt The source statement to be converted.
 * \return The converted form.
 */
Stmt ConvertSSA(Stmt stmt);

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
Stmt Inline(FunctionRef f,
            Array<Var> args,
            Expr body,
            Stmt stmt);

/*!
 * \brief Schedule s' dependent operations.
 *
 * \param s The schedule to be realized
 * \return the result Stmt
 */
Stmt ScheduelOps(Schedule s);

}  // namespace ir
}  // namespace tvm

#endif  // TVM_IR_PASS_H_
