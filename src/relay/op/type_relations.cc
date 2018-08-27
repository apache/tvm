/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_relations.cc
 * \brief A set of utilities and common functionality
 * for type relations.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include "../pass/incomplete_type.h"

namespace tvm {
namespace relay {

TensorType as_ttype(const Type & t) {
  if (auto tt_node = t.as<TensorTypeNode>()) {
    return GetRef<TensorType>(tt_node);
  } else {
    return TensorType(nullptr);
  }
}

Array<Type> IdentityRel(const Array<Type> & types, int num_args) {
    CHECK(types.size() == 1);
    auto t1 = as_ttype(types[0]);
    if (t1 && types[1].as<IncompleteTypeNode>()) {
        return {t1, t1};
    } else {
        return types;
    }
}

Array<Type> BroadcastRel(const Array<Type> & types, int num_args) {
  std::cout << "Inside of Broadcast" << std::endl;
  CHECK(types.size() == 0);
  if (auto t1 = as_ttype(types[0])) {
    if (auto t2 = as_ttype(types[1])) {
      return types;
    }
  }
  return types;
}


}  // namespace relayv
}  // namespace tvm
