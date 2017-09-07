/*!
 *  Copyright (c) 2017 by Contributors
 * \file message_passing.h
 * \brief Common utilities to do message passing
 *  on the schedule hyper graph.
 */
#ifndef TVM_SCHEDULE_MESSAGE_PASSING_H_
#define TVM_SCHEDULE_MESSAGE_PASSING_H_

#include <tvm/expr.h>
#include <tvm/schedule.h>
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace schedule {
/*!
 * \brief Downward inference of domain of each IterVar.
 *  Caller set the range of the root, then the function
 *  propagates it towards the leaves.
 *
 * \param stage The stage to operate on.
 * \param p_state The state of the message passing.
 * \param allow_missing Whether allow missing value.
 */
void PassDownDomain(
    const Stage& stage,
    std::unordered_map<IterVar, Range>* p_state,
    bool allow_missing = false);

/*!
 * \param Upward inference of index of each IterVar.
 *  given index assignement of the leaves,
 *
 * \param stage The stage to operate on.
 * \param dom_map The domain map of each iteration variable's domain.
 * \param p_state The index state of each IterVar.
 * \param allow_missing Whether allow missing value.
 */
void PassUpIndex(const Stage& stage,
                 const Map<IterVar, Range>& dom_map,
                 std::unordered_map<IterVar, Expr>* p_state,
                 bool allow_missing = false);

/*!
 * \param Downward inference of index of each IterVar.
 *  given index assignement of roots.
 *
 * \param stage The stage to operate on.
 * \param dom_map The domain map of each iteration variable's domain.
 * \param p_state The index state of each IterVar.
 * \param allow_missing Whether allow missing value.
 */
void PassDownIndex(const Stage& stage,
                   const Map<IterVar, Range>& dom_map,
                   std::unordered_map<IterVar, Expr>* p_state,
                   bool allow_missing = false);

/*!
 * \param Upward inference of domain set of each IterVar.
 *  given domain assignment of the leaves,
 *
 * \param stage The stage to operate on.
 * \param dom_map The domain map of each iteration variable's maximum domain.
 * \param p_state The index state of each IterVar.
 */
void PassUpDomain(const Stage& stage,
                  const std::unordered_map<IterVar, Range>& dom_map,
                  std::unordered_map<IterVar, IntSet>* p_state);

/*!
 * \brief Upward message passing of bitmask with or relation.
 * \param stage The stage to operate on.
 * \param p_state The index state of each IterVar.
 * \param allow_missing Whether allow missing value.
 */
void PassUpBitMaskOr(const Stage& stage,
                     std::unordered_map<IterVar, int>* p_state,
                     bool allow_missing = false);

/*!
 * \brief Downward message passing of bitmask with or relation.
 * \param stage The stage to operate on.
 * \param p_state The index state of each IterVar.
 * \param allow_missing Whether allow missing value.
 */
void PassDownBitMaskOr(const Stage& stage,
                       std::unordered_map<IterVar, int>* p_state,
                       bool allow_missing = false);

/*!
 * \brief Create boundary check predicates given remapped value of root
 * \param stage The stage we operate on
 * \param dom_map The domain map of each value.
 * \param value_map The value map of the root iter var.
 * \param skip_ivar_domain Whether we skip check for IterVar's original domain.
 * \param skip_iter The set of variables to skip bound condition.
 * \return List of predicates that we need to check.
 */
std::vector<Expr>
MakeBoundCheck(
    const Stage& stage,
    const Map<IterVar, Range>& dom_map,
    const std::unordered_map<IterVar, Expr>& value_map,
    bool skip_ivar_domain,
    const std::unordered_set<IterVar>& skip_iter);

}  // namespace schedule
}  // namespace tvm
#endif  // TVM_SCHEDULE_MESSAGE_PASSING_H_
