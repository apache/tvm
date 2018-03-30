/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Helper utilities to implement compute_op.
 * \file compute_op.h
 */
#ifndef TVM_OP_COMPUTE_OP_H_
#define TVM_OP_COMPUTE_OP_H_

#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <vector>
#include <unordered_map>

namespace tvm {
// loop nest structure for general compute
// This the the loop nest structured used in compute.
// Does not include the loop body.
struct ComputeLoopNest {
  // The common number of loops between init and main
  size_t num_common_loop;
  // predicates for the initialize loop
  std::vector<Expr> init_predicates;
  // Initialization nest involved.
  std::vector<std::vector<Stmt> > init_nest;
  // Value map for the init code
  std::unordered_map<IterVar, Expr> init_vmap;
  // Predicates for the main update loop
  std::vector<Expr> main_predicates;
  // The general loop nest
  std::vector<std::vector<Stmt> > main_nest;
  // Value map for the IterVar.
  std::unordered_map<IterVar, Expr> main_vmap;

  /*!
   * \brief constructor to build ComputeOpNest
   * \param self The pointer to compute op.
   * \param stage The scxhedule stage.
   * \param dom_map The domain map.
   * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
   * \return The constructed loop nest
   */
  static ComputeLoopNest make(
      const ComputeOpNode* self,
      const Stage& stage,
      const std::unordered_map<IterVar, Range>& dom_map,
      bool debug_keep_trivial_loop);
};

/*!
 * \brief Build body of compute for cross thread reduction pattern.
 * \param self The pointer to ComputeOpNode
 * \param stage The schedule stage.
 * \param dom_map The domain map.
 * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
 * \return The created statement.
 */
Stmt MakeCrossThreadReduction(
    const ComputeOpNode* self,
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop);

/*!
 * \brief Build body of compute for tensorization.
 * \param self The pointer to ComputeOpNode
 * \param stage The schedule stage.
 * \param dom_map The domain map.
 * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
 * \return The created statement.
 */
Stmt MakeTensorize(const ComputeOpNode* self,
                   const Stage& stage,
                   const std::unordered_map<IterVar, Range>& dom_map,
                   bool debug_keep_trivial_loop);
}  // namespace tvm

#endif  // TVM_OP_COMPUTE_OP_H_
