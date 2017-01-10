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
Map<IterVar, Range> InferBound(Schedule sch);

}  // namespace schedule
}  // namespace tvm
#endif  // TVM_SCHEDULE_PASS_H_
