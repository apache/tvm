/*!
 *  Copyright (c) 2016 by Contributors
 * \file bound.h
 * \brief The bound inference logics on the schedule.
 */
#ifndef TVM_BOUND_BOUND_H_
#define TVM_BOUND_BOUND_H_

#include <tvm/expr.h>
#include <tvm/schedule.h>
#include <unordered_map>

namespace tvm {
namespace bound {

/*!
 * \brief Infer the bound of all iteration variables relates to the schedule.
 *
 * \param sch The root schedule to infer all the bounds.
 * \return the result bound of the iteration Variable
 */
std::unordered_map<IterVar, Range> InferBound(Schedule sch);

}  // namespace bound
}  // namespace tvm

#endif  // TVM_BOUND_BOUND_H_
