/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/op/type_relations.h
 * \brief A set of utilities and common functionality
 * for type relations.
 */
#ifndef TVM_RELAY_OP_TYPE_RELATIONS_H_
#define TVM_RELAY_OP_TYPE_RELATIONS_H_

#include <tvm/relay/type.h>
#include <string>

namespace tvm {
namespace relay {

Array<Type> IdentityRel(const Array<Type> & types, int num_args);
Array<Type> BroadcastRel(const Array<Type> & types, int num_args);
Array<Type> BroadcastCompRel(const Array<Type> & types, int num_args);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_TYPE_RELATIONS_H_
