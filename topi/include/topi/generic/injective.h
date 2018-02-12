/*!
*  Copyright (c) 2017 by Contributors
* \file generic/injective.h
* \brief Generic schedule for injective operations
*/
#ifndef TOPI_GENERIC_INJECTIVE_H_
#define TOPI_GENERIC_INJECTIVE_H_

#include "topi/tags.h"
#include "topi/detail/fuse.h"
#include "tvm/tvm.h"
#include "tvm/build_module.h"

namespace topi {
using namespace tvm;

namespace generic {

/*!
 * \brief Create a generic schedule for the given injective ops.
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule schedule_injective(const Target &target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  tvm::schedule::AutoInlineInjective(s);
  auto x = outs[0];
  detail::Fuse(s[x], s[x]->op.as<ComputeOpNode>()->axis);

  return s;
}

}  // namespace generic
}  // namespace topi
#endif  // TOPI_GENERIC_INJECTIVE_H_
