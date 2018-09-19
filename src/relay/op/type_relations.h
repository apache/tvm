/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/op/type_relations.h
 * \brief A set of utilities and common functionality
 * for type relations.
 */
#ifndef TVM_RELAY_OP_TYPE_RELATIONS_H_
#define TVM_RELAY_OP_TYPE_RELATIONS_H_

#include <tvm/relay/error.h>
#include <tvm/relay/type.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief The error raised by a type relation.
 *
 * This error is how a type relation signals that it has failed.
 *
 */
struct TypeRelationError : Error {
  explicit TypeRelationError(const std::string& msg)
      : Error(msg) {}
};

/*! \brief The identity type relation maps a single input variable
 * to the output variable.
 *
 * \param types The input and output types to the relation.
 * \param num_args The number of input arguments.
 * \return The (potentially partial) solution to the relation.
 */
Array<Type> IdentityRel(const Array<Type>& types, int num_args);
/*! \brief The broadcast type relation, implements the broadcasting
 * rule over the two input types producing the broadcasted type.
 *
 * \param types The input and output types to the relation.
 * \param num_args The number of input arguments.
 * \return The (potentially partial) solution to the relation.
 */
Array<Type> BroadcastRel(const Array<Type>& types, int num_args);
/*! \brief The broadcast type relation, implements the broadcasting
 * rule over the two input types producing the broadcasted type.
 *
 * This differs from BroadcastRel in the return dtype,
 * it instead returns bool, for use in comparsion operators
 * such as equal, not_equal, lt, and so on.
 *
 * \param types The input and output types to the relation.
 * \param num_args The number of input arguments.
 * \return The (potentially partial) solution to the relation.
 */
Array<Type> BroadcastCompRel(const Array<Type>& types, int num_args);

/*! \brief The concat relation.
 *
 * This relation takes a single input which must be a single tensor
 * or an arbitrary sized tuple. It combines these input dimensions
 * together to produce the output example.
 */
Array<Type> ConcatRel(const Array<Type>& types, int num_args);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_TYPE_RELATIONS_H_
