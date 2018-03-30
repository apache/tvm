/*!
*  Copyright (c) 2017 by Contributors
* \file rocm/dense.h
* \brief rocm schedule for dense operation
*/
#ifndef TOPI_ROCM_DENSE_H_
#define TOPI_ROCM_DENSE_H_

#include "tvm/tvm.h"
#include "tvm/build_module.h"
#include "topi/tags.h"
#include "topi/detail/array_utils.h"
#include "topi/nn/dense.h"
#include "topi/contrib/rocblas.h"
#include "topi/generic/extern.h"
#include "topi/cuda/dense.h"

namespace topi {
using namespace tvm;

namespace rocm {
/*!
* \brief Implementation of dense for rocm backend
*
* \param target The target device
* \param data Tensor with shape [batch, in_dim]
* \param weight Tensor with shape [out_dim, in_dim]
* \param bias Tensor with shape [out_dim]. Optional; to omit bias, pass Tensor()
*
* \return Tensor with shape [batch, out_dim]
*/
inline tvm::Tensor dense_rocm(const Target& target,
                              const tvm::Tensor& data,
                              const tvm::Tensor& weight,
                              const tvm::Tensor& bias) {
  CHECK_EQ(data->shape.size(), 2) << "dense requires 2-D data";
  CHECK_EQ(weight->shape.size(), 2) << "dense requires 2-D weight";
  if (bias.defined()) {
    CHECK_EQ(bias->shape.size(), 1) << "dense requires 1-D bias";
  }

  auto batch = data->shape[0];
  auto in_dim = data->shape[1];
  auto out_dim = weight->shape[0];

  if (target->libs().count("rocblas")) {
    auto mm = topi::contrib::rocblas_matmul(data, weight, false, true);
    if (bias.defined()) {
      mm = tvm::compute({ batch, out_dim },
                        [&](Var i, Var j) {
                          return mm(i, j) + bias(j);
                        }, "tensor", kBroadcast);
    }

    return mm;
  } else {
    return topi::nn::dense(data, weight, bias);
  }
}

/*!
* \brief Create a rocm schedule for dense
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule schedule_dense(const Target &target, const Array<Tensor>& outs) {
  if (target->target_name == "rocm" &&
    target->libs().count("rocblas")) {
    return topi::generic::schedule_extern(target, outs);
  }

  return topi::cuda::schedule_dense(target, outs);
}

}  // namespace rocm
}  // namespace topi
#endif  // TOPI_ROCM_DENSE_H_

