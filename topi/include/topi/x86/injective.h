/*!
*  Copyright (c) 2017 by Contributors
* \file x86/injective.h
* \brief x86 schedule for injective ops
*/
#ifndef TOPI_X86_INJECTIVE_H_
#define TOPI_X86_INJECTIVE_H_

#include "topi/tags.h"
#include "topi/detail/fuse.h"
#include "tvm/tvm.h"
#include "tvm/build_module.h"

namespace topi {
using namespace tvm;

namespace x86 {
/*!
* \brief Create an x86 schedule for the given injective ops.
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
  auto axis = s[x]->op.as<ComputeOpNode>()->axis;
  if (axis.size() == 4) {
    auto n = axis[0];
    auto c = axis[1];
    auto fused = detail::Fuse(s[x], { n, c });  // for nhwc layout, fuse n and h
    s[x].parallel(fused);
  } else {
    s[x].parallel(axis[0]);
  }

  return s;
}

}  // namespace x86
}  // namespace topi
#endif  // TOPI_X86_INJECTIVE_H_
