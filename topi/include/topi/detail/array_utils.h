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

}  // namespace detail
}  // namespace topi
#endif  // TOPI_DETAIL_ARRAY_UTILS_H_
