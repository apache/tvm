/*!
 *  Copyright (c) 2018 by Contributors
 * \file op_common.h
 * \brief A set of utilities and common functionality
 * for relay ops.
 */
#ifndef TVM_RELAY_OP_OP_COMMON_H_
#define TVM_RELAY_OP_OP_COMMON_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <vector>

namespace tvm {
namespace relay {

template<typename T>
std::vector<T> AsVector(const Array<T> &array) {
    std::vector<T> result;
    result.reserve(array.size());
    for (const T& ele : array) {
        result.push_back(ele);
    }
    return result;
}

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_OP_COMMON_H_
