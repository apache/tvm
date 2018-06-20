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
* \brief Helper to create a default x86 schedule for the given ops.
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
* \param auto_inline Whether to apply the auto inline step.
*
* \return A schedule for the given ops.
*/
inline Schedule MakeDefaultSchedule(const Target &target,
                                    const Array<Tensor>& outs,
                                    bool auto_inline) {
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
      detail::Fuse(s[x], axis);
    }
    return s;
  }

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

/*!
* \brief Create a default x86 schedule for the given ops.
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule default_schedule(const Target &target, const Array<Tensor>& outs) {
  return MakeDefaultSchedule(target, outs, false);
}

/*!
* \brief Create a default x86 schedule for the given ops, with auto inline
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule default_schedule_auto_inline(const Target &target, const Array<Tensor>& outs) {
  return MakeDefaultSchedule(target, outs, true);
}

}  // namespace x86
}  // namespace topi
#endif  // TOPI_X86_DEFAULT_H_
