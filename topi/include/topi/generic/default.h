/*!
*  Copyright (c) 2017 by Contributors
* \file generic/default.h
* \brief Generic default schedule
*/
#ifndef TOPI_GENERIC_DEFAULT_H_
#define TOPI_GENERIC_DEFAULT_H_

#include "topi/tags.h"
#include "topi/detail/fuse.h"
#include "tvm/tvm.h"
#include "tvm/build_module.h"

namespace topi {
using namespace tvm;

namespace generic {
/*!
* \brief Create a generic default schedule for the given output tensors.
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule default_schedule(const Target& target, Array<Tensor> outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  return s;
}

/*!
* \brief Create a generic default schedule for the given output tensors, and apply
* auto inline
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule default_schedule_auto_inline(const Target& target, Array<Tensor> outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  auto x = outs[0];
  tvm::schedule::AutoInlineInjective(s);
  auto axis = s[x]->op.as<ComputeOpNode>()->axis;
  if (axis.size() > 0) {
    detail::Fuse(s[x], axis);
  }
  return s;
}

}  // namespace generic
}  // namespace topi
#endif  // TOPI_GENERIC_DEFAULT_H_
