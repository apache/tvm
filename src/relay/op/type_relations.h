/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/op/type_relations.h
 * \brief A set of utilities and common functionality
 * for type relations.
 */
#ifndef TVM_RELAY_TYPECK_RESOLVE_H_
#define TVM_RELAY_TYPECK_RESOLVE_H_

#include <string>
#include <tvm/relay/type.h>

namespace tvm {
namespace relay {

Array<Type> IdentityRel(const Array<Type> & types, int num_args);
Array<Type> BroadcastRel(const Array<Type> & types, int num_args);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TYPECK_RESOLVE_H_
