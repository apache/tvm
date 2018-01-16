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
using namespace tvm;

/*! \brief Get padding size for each side given padding height and width */
std::vector<int> GetPadTuple(int pad_h, int pad_w) {
  auto pad_top = (pad_h + 1) / 2;
  auto pad_left = (pad_w + 1) / 2;

  return { pad_top, pad_left, pad_h - pad_top, pad_w - pad_left };
}

}  // namespace topi
#endif  // TOPI_DETAIL_PAD_UTILS_H_
