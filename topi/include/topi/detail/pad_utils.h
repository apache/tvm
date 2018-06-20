/*!
*  Copyright (c) 2017 by Contributors
* \file pad_utils.h
* \brief Padding helpers
*/
#ifndef TOPI_DETAIL_PAD_UTILS_H_
#define TOPI_DETAIL_PAD_UTILS_H_

#include <vector>

#include "tvm/tvm.h"

namespace topi {
namespace detail {
using namespace tvm;

/*!
 * \brief Get padding size for each side given padding height and width
 *
 * \param pad_h The amount to pad each of the top and bottom sides
 * \param pad_w The amount to pad each of the left and right sides
 *
 * \return An array of 4 elements, representing padding sizes for
 * each individual side. The array is in the order { top, left, bottom, right }
 */
inline Array<Expr> GetPadTuple(Expr pad_h, Expr pad_w) {
  pad_h *= 2;
  pad_w *= 2;

  auto pad_top = (pad_h + 1) / 2;
  auto pad_left = (pad_w + 1) / 2;

  return { pad_top, pad_left, pad_h - pad_top, pad_w - pad_left };
}

}  // namespace detail
}  // namespace topi
#endif  // TOPI_DETAIL_PAD_UTILS_H_
