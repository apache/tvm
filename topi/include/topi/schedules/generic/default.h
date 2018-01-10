/*!
*  Copyright (c) 2017 by Contributors
* \file default.h
* \brief Generic default schedule
*/
#ifndef TOPI_SCHEDULES_GENERIC_DEFAULT_H_
#define TOPI_SCHEDULES_GENERIC_DEFAULT_H_

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
 * \param auto_inline Whether to apply the auto inline step.
 *
 * \return A schedule for the given ops.
 */
Schedule default_schedule(const Target& target, Array<Tensor> outs, bool auto_inline) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  if (auto_inline) {
    auto x = outs[0];
    tvm::schedule::AutoInlineInjective(s);
    Fuse(s[x], s[x]->op.as<ComputeOpNode>()->axis);
  }
  return s;
}

}  // namespace generic
}  // namespace topi
#endif  // TOPI_SCHEDULES_GENERIC_DEFAULT_H_
