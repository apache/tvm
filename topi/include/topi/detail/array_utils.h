/*!
*  Copyright (c) 2017 by Contributors
* \file array_utils.h
* \brief Utility functions for handling arrays
*/
#ifndef TOPI_DETAIL_ARRAY_UTILS_H_
#define TOPI_DETAIL_ARRAY_UTILS_H_

#include "tvm/tvm.h"

namespace topi {
namespace detail {
using namespace tvm;

/*!
 * \brief Search an array for a specific item
 *
 * \param array The array to search
 * \param item The item to search for
 *
 * \return True iff the given array contains the given item.
 */
template<typename T>
inline bool contains(Array<T> array, T item) {
  for (auto& i : array) {
    if (i == item) {
      return true;
    }
  }
  return false;
}

/*!
 * \brief Returns a positive axis index. Panics if index is out of bounds.
 *
 * \param ndim The number of array dimensions.
 * \param axis The axis; can be negative.
 *
 * \return a number in [0, ndim) which identifies the axis.
 */
inline size_t getRealAxis(int ndim, int axis) {
  int realAxis = axis >= 0 ? axis : axis + ndim;
  CHECK(realAxis >= 0) << "Axis " << axis << " out of bounds for " << ndim << "D array.";
  return realAxis;
}

}  // namespace detail
}  // namespace topi
#endif  // TOPI_DETAIL_ARRAY_UTILS_H_
