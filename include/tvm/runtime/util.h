/*!
 *  Copyright (c) 2017 by Contributors
 * \file util.h
 * \brief Useful runtime util.
 */
#ifndef TVM_RUNTIME_UTIL_H_
#define TVM_RUNTIME_UTIL_H_

#include "./c_runtime_api.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Check whether type matches the given spec.
 * \param t The type
 * \param code The type code.
 * \param bits The number of bits to be matched.
 * \param lanes The number of lanes sin the type.
 */
inline bool TypeMatch(TVMType t, int code, int bits, int lanes = 1) {
  return t.code == code && t.bits == bits && t.lanes == lanes;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_UTIL_H_
