/*!
 *  Copyright (c) 2017 by Contributors
 * \file op_util.h
 * \brief Common utility used in operator construction.
 */
#ifndef TVM_OP_OP_UTIL_H_
#define TVM_OP_OP_UTIL_H_

#include <tvm/expr.h>
#include <tvm/schedule.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../pass/ir_util.h"

namespace tvm {
namespace op {

using ir::MergeNest;

/*!
 * \brief Build loop nest for stage.
 *
 * \param stage The stage to create a loop nest.
 * \param dom_map The range of each iter var.
 * \param begin_iter_pos The beginning position of leaf_iter_vars to generate loop.
 * \param new_loop_var Whether create new loop variable.
 * \param skip_iter Whether skip certain iteration.
 * \param p_value_map The result value of each IterVar.
 */
std::vector<std::vector<Stmt> >
MakeLoopNest(const Stage& stage,
             const std::unordered_map<IterVar, Range>& dom_map,
             size_t begin_iter_pos,
             bool new_loop_var,
             const std::unordered_set<IterVar>& skip_iter,
             std::unordered_map<IterVar, Expr>* p_value_map);
/*!
 * \brief Create boundary check condition for given stage.
 *
 * \param stage The stage to create a loop nest.
 * \param dom_map The range of each iter var.
 * \param skip_ivar_domain Whether we can skip check for IterVar's original domain.
 * \param skip_iter Whether skip certain iteration.
 * \param value_map The result value of each IterVar.
 * \return List of predicates that we need to check.
 */
std::vector<Expr>
MakeBoundCheck(const Stage& stage,
               const Map<IterVar, Range>& dom_map,
               bool skip_ivar_domain,
               const std::unordered_set<IterVar>& skip_iter,
               const std::unordered_map<IterVar, Expr>& value_map);
/*!
 * \brief Create a nest of if checking the predicates.
 *
 * \param predicates The predicates to be checked.
 * \return List of If nest that checks the predicates.
 */
std::vector<Stmt> MakeIfNest(const std::vector<Expr>& predicates);

/*!
 * \brief Replace the tensor reference in stmt by the replace map.
 * \param stmt The statement to be processed.
 * \param replace The replacement rule.
 */
Stmt ReplaceTensor(Stmt stmt,
                   const std::unordered_map<Tensor, Tensor>& replace);
/*!
 * \brief Replace the tensor reference in expr by the replace map.
 * \param expr The expression to be processed.
 * \param replace The replacement rule.
 */
Expr ReplaceTensor(Expr expr,
                   const std::unordered_map<Tensor, Tensor>& replace);

}  // namespace op
}  // namespace tvm
#endif  // TVM_OP_OP_UTIL_H_
