/*!
*  Copyright (c) 2017 by Contributors
* \file array_utils.h
* \brief Utility functions for handling arrays
*/
#ifndef TOPI_DETAIL_ARRAY_UTILS_H_
#define TOPI_DETAIL_ARRAY_UTILS_H_

#include "tvm/tvm.h"

namespace topi {
using namespace tvm;

/*! \brief returns true iff the given array contains the given item */
template<typename T>
bool contains(Array<T> array, T item) {
  return std::any_of(array.begin(), array.end(),
    [&](T i) { return i == item; });
}

}  // namespace topi
#endif  // TOPI_DETAIL_ARRAY_UTILS_H_
