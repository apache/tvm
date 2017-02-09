/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_pass.h
 * \brief Collection of IR pass functions
 *
 *  When the pass functions in this file are for Stmt,
 *  we can use PassFunction(Evaluate(expr)) to apply it to Expr
 */
#ifndef TVM_IR_PASS_H_
#define TVM_IR_PASS_H_

#include <ir/IREquality.h>
#include <pass/Simplify.h>
#include <tvm/ir_functor.h>
#include <unordered_map>
#include <vector>
#include <string>
#include "./expr.h"
#include "./buffer.h"
#include "./schedule.h"
#include "./lowered_func.h"

namespace tvm {
namespace ir {

inline bool Equal(Expr a, Expr b) {
  return Halide::Internal::equal(a, b);
}

inline bool Equal(Stmt a, Stmt b) {
  return Halide::Internal::equal(a, b);
}

inline Expr Simplify(Expr a) {
  return Halide::Internal::simplify(a);
}

inline Stmt Simplify(Stmt a) {
  return Halide::Internal::simplify(a);
}

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
 * \brief Whether the expression have side effect.
 * \return whether expression have side effect
 */
bool HasSideEffect(const Expr& e);

/*!
 * \brief Convert a IR node to be SSA form.
 * \param stmt The source statement to be converted.
 * \return The converted form.
 */
Stmt ConvertSSA(Stmt stmt);

/*!
 * \brief Simplify by applying canonical form.
 * \param stmt The statement to be canonically simplifed.
 * \return Canonicalized statement.
 */
Stmt CanonicalSimplify(Stmt stmt);

/*!
 * \brief Substitute the var specified in key->var to be value.
 * \param stmt The source statement to be substituted
 * \param value_map The map of new values.
 * \return The converted form.
 */
Stmt Substitute(Stmt stmt, const Map<Var, Expr>& value_map);

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
Stmt Inline(Stmt stmt,
            FunctionRef f,
            Array<Var> args,
            Expr body);

/*!
 * \brief Flatten the multi-dimensional read/write
 *  to single dimensional Load/Store
 *
 * \param stmt The stmt to be trasnformed.
 * \param extern_buffer Map specifies external
 *    buffer assignment of input and outputs.
 */
Stmt StorageFlatten(Stmt stmt,
                    Map<Tensor, Buffer> extern_buffer);

/*!
 * \brief unroll the constant loops
 * \param stmt The statment to be unrolled.
 * \param max_auto_step The maximum step to stop performing automatic unrolling.
 */
Stmt UnrollLoop(Stmt stmt, int max_auto_step);

/*!
 * \brief vectorize the constant loops
 * \param stmt The statment to be vectorized.
 */
Stmt VectorizeLoop(Stmt stmt);

/*!
 * \brief Make an user callable API LoweredFunc.
 *
 *  The main task of this function is to create code to :
 *   - Map the values in the api_args to of Var that is required by body.
 *   - Insert assertions to check type/value of the passed arguments.
 *
 * \param body The body of the function.
 * \param name The name of the function.
 * \param api_args Arguments to the function, can be either Var, or Buffer
 * \param num_packed_args Number of arguments that are processed in packed form.
 * \return a LoweredFunc with the specified signiture.
 *
 * \note
 *  The function signiture have two cases
 *
 *  if num_packed_args is zero:
 *     f(api_arg_0, api_arg_1, .., api_arg_n) where n == len(api_args)
 *
 *  if num_packed_args is not zero:
 *       f(TVMArg* packed_args, int* packed_arg_type_ids, int num_packed_args,
 *         api_arg_k, api_arg_k+1, ... api_arg_n)
 *
 *       where n == len(api_args), k == num_packed_args
 *
 *  There is no thread_axis in generated function.
 */
LoweredFunc MakeAPI(Stmt body,
                    std::string name,
                    Array<NodeRef> api_args,
                    int num_packed_args);

/*!
 * \brief Count number of undefined vars in f.
 * \param f The function to be checked.
 * \return Number of undefined vars.
 */
Array<Var> UndefinedVars(const LoweredFunc& f);

/*!
 * \brief Split the function into a host function and device functions.
 * \param func The function to be splitted.
 *
 * \return Array of functions, the first one is host function,
 *     the others are device functions.
 */
Array<LoweredFunc> SplitHostDevice(LoweredFunc func);

/*!
 * \brief Insert sync between parallel read/write of shared buffers.
 *
 * \param stmt The stmt to be trasnformed.
 * \param storage_scope The storage scope considered.
 */
LoweredFunc StorageSync(LoweredFunc stmt, std::string storage_scope);

}  // namespace ir
}  // namespace tvm

#endif  // TVM_IR_PASS_H_
