/*!
*  Copyright (c) 2017 by Contributors
* \file generic/extern.h
* \brief Schedule for extern followed by injective ops
*/
#ifndef TOPI_GENERIC_EXTERN_H_
#define TOPI_GENERIC_EXTERN_H_

#include "topi/tags.h"
#include "topi/detail/fuse.h"
#include "tvm/tvm.h"
#include "tvm/build_module.h"

namespace topi {
using namespace tvm;

namespace generic {
/*!
* \brief Schedule an extern op followed by injective operations
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the op.
*/
inline Schedule schedule_extern(const Target& target, Array<Tensor> outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  return s;
}

}  // namespace generic
}  // namespace topi
#endif  // TOPI_GENERIC_EXTERN_H_
