/*!
 *  Copyright (c) 2016 by Contributors
 * \file bound.h
 * \brief The bound inference logics on the schedule.
 */
#ifndef TVM_SCHEDULE_BOUND_H_
#define TVM_SCHEDULE_BOUND_H_

#include <tvm/expr.h>
#include <tvm/schedule.h>
#include <unordered_map>

namespace tvm {
namespace schedule {

/*!
 * \brief Infer the bound of all iteration variables relates to the schedule.
 *
 * \param sch The root schedule to infer all the bounds.
 * \return the result bound of the iteration Variable
 */
std::unordered_map<IterVar, Range> InferBound(Schedule sch);

}  // namespace schedule
}  // namespace tvm

#endif  // TVM_SCHEDULE_BOUND_H_
