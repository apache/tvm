/*!
*  Copyright (c) 2017 by Contributors
* \file x86/default.h
* \brief default x86 schedule
*/
#ifndef TOPI_X86_DEFAULT_H_
#define TOPI_X86_DEFAULT_H_

#include "topi/tags.h"
#include "topi/detail/fuse.h"
#include "tvm/tvm.h"
#include "tvm/build_module.h"

namespace topi {
using namespace tvm;

namespace x86 {
/*!
* \brief Create a default x86 schedule for the given ops.
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
* \param auto_inline Whether to apply the auto inline step.
*
* \return A schedule for the given ops.
*/
Schedule default_schedule(const Target &target, const Array<Tensor>& outs, bool auto_inline) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  auto x = outs[0];
  auto axis = s[x]->op.as<ComputeOpNode>()->axis;

  if (auto_inline) {
    tvm::schedule::AutoInlineInjective(s);
    if (axis.size() > 0) {
      Fuse(s[x], axis);
    }
    return s;
  }

  if (axis.size() == 4) {
    auto n = axis[0];
    auto c = axis[1];
    auto fused = Fuse(s[x], { n, c });  // for nhwc layout, fuse n and h
    s[x].parallel(fused);
  } else {
    s[x].parallel(axis[0]);
  }

  return s;
}

}  // namespace x86
}  // namespace topi
#endif  // TOPI_X86_DEFAULT_H_
