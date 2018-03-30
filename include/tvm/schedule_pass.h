/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_pass.h
 * \brief  Collection of Schedule pass functions.
 *
 *  These passes works on the schedule hyper-graph
 *  and infers information such as bounds, check conditions
 *  read/write dependencies between the IterVar
 */
#ifndef TVM_SCHEDULE_PASS_H_
#define TVM_SCHEDULE_PASS_H_

#include "./base.h"
#include "./schedule.h"

namespace tvm {
namespace schedule {

/*!
 * \brief Infer the bound of all iteration variables relates to the schedule.
 *
 * \param sch The root schedule to infer all the bounds.
 * \return the result bound of the iteration Variable
 */
Map<IterVar, Range> InferBound(const Schedule& sch);

/*!
 * \brief Schedule s' dependent operations.
 *
 * \param s The schedule to be realized
 * \param dom_map The domain of each iter vars.
 * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1 during lowering.
 *                                This is a debug feature for dataflow/axis analysis.
 *                                Note: If this is true, The lowered IR may be incorrect,
 *                                because we will also delete the init part of reduction
 * \return the result Stmt
 */
Stmt ScheduleOps(Schedule s, Map<IterVar, Range> dom_map, bool debug_keep_trivial_loop);

/*!
 * \brief To automatically inline the element-wise operations.
 *
 * \param sch The schedule to be inlined.
 */
void AutoInlineElemWise(Schedule sch);

/*!
 * \brief To automatically inline operations with injective writes
 *   (i.e. writes without reduction or sequential loops). Note
 *   that in this case, guarantees about contiguity, transpose, stride,
 *   alignemnt and memory footprint in general do not hold.
 *
 * \param sch The schedule to be inlined.
 */
EXPORT void AutoInlineInjective(Schedule sch);

}  // namespace schedule
}  // namespace tvm
#endif  // TVM_SCHEDULE_PASS_H_
