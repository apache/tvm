/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external generic C library function
 */

#ifndef TVM_CONTRIB_GENERIC_UTILS_H_
#define TVM_CONTRIB_GENERIC_UTILS_H_

#include <tvm/runtime/registry.h>

namespace tvm {
namespace contrib {

template<typename DType>
struct SortElem {
  DType value;
  int32_t index;
  static bool is_descend;

  SortElem(DType v, int32_t i) {
    value = v;
    index = i;
  }

  bool operator<(const SortElem &other) const {
    return is_descend ? value > other.value : value < other.value;
  }
};

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_GENERIC_UTILS_H_
