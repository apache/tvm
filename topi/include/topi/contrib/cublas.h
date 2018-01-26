/*!
 *  Copyright (c) 2017 by Contributors
 * \brief External function interface to cuBLAS libraries
 * \file tags.h
 */
#ifndef TOPI_CONTRIB_CUBLAS_H_
#define TOPI_CONTRIB_CUBLAS_H_

#include "tvm/tvm.h"
#include "topi/detail/extern.h"

namespace topi {
namespace contrib {
using namespace tvm;
using namespace topi::detail;
/*!
* \brief Create an op that multiplies lhs and rhs with cuBLAS
*
* \param lhs The left matrix operand
* \param rhs The right matrix operand
* \param transa Whether to transpose lhs
* \param transb Whether to transpose rhs
*
* \return The output tensor
*/
inline Tensor cublas_matmul(const Tensor& lhs,
                            const Tensor& rhs,
                            bool transa,
                            bool transb) {
  auto n = transa ? lhs->shape[1] : lhs->shape[0];
  auto m = transb ? rhs->shape[0] : rhs->shape[1];

  return make_extern(
    { { n, m } }, { lhs->dtype }, { lhs, rhs },
    [&](Array<Buffer> ins, Array<Buffer> outs) {
      return call_packed({
        Expr("tvm.contrib.cublas.matmul"),
        pack_buffer(ins[0]),
        pack_buffer(ins[1]),
        pack_buffer(outs[0]),
        transa,
        transb });
    }, "C", "")[0];
}

}  // namespace contrib
}  // namespace topi

#endif  // TOPI_CONTRIB_CUBLAS_H_
