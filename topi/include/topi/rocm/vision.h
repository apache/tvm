/*!
*  Copyright (c) 2018 by Contributors
* \file rocm/vision.h
* \brief rocm schedule for region operation
*/
#ifndef TOPI_ROCM_VISION_H_
#define TOPI_ROCM_VISION_H_

#include "tvm/tvm.h"
#include "tvm/build_module.h"
#include "topi/tags.h"
#include "topi/detail/array_utils.h"
#include "topi/contrib/rocblas.h"
#include "topi/generic/extern.h"
#include "topi/cuda/vision.h"

namespace topi {
using namespace tvm;
namespace rocm {
/*!
* \brief Create a rocm schedule for region
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule schedule_region(const Target &target, const Array<Tensor>& outs) {
  return topi::cuda::schedule_region(target, outs);
}
}  // namespace rocm
}  // namespace topi
#endif  // TOPI_ROCM_VISION_H_
