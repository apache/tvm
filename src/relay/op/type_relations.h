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

/*!
 * \brief The identity type relation, all the types are equal.
 *
 * \param types The input and output types to the relation.
 * \param num_inputs The number of input arguments.
 * \param attrs The attributes
 * \param reporter The reporter.
 * \return true whether relation has been resolved.
 */
bool IdentityRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
                 const TypeReporter& reporter);

/*!
 * \brief The broadcast type relation, implements the broadcasting
 * rule over the two input types producing the broadcasted type.
 *
 * \param types The input and output types to the relation.
 * \param num_inputs The number of input arguments.
 * \param attrs The attributes
 * \param reporter The reporter.
 * \return true whether relation has been resolved.
 */
bool BroadcastRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter);

/*!
 * \brief The broadcast type relation, implements the broadcasting
 *  rule over the two input types producing the broadcasted type.
 *
 * This differs from BroadcastRel in the return dtype,
 * it instead returns bool(uint8), for use in comparsion operators
 * such as equal, not_equal, lt, and so on.
 *
 * \param types The input and output types to the relation.
 * \param num_inputs The number of input arguments.
 * \param attrs The attributes
 * \param reporter The reporter.
 * \return true whether relation has been resolved.
 */
bool BroadcastCompRel(const Array<Type>& types,
                      int num_inputs,
                      const Attrs& attrs,
                      const TypeReporter& reporter);

/*!
 * \brief The The concat relation, implements the broadcasting
 *  rule over the two input types producing the broadcasted type.
 *
 * This differs from BroadcastRel in the return dtype,
 * it instead returns bool(uint8), for use in comparsion operators
 * such as equal, not_equal, lt, and so on.
 *
 * \param types The input and output types to the relation.
 * \param num_inputs The number of input arguments.
 * \param attrs The attributes
 * \param reporter The reporter.
 * \return true whether relation has been resolved.
 */
bool ConcatRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_TYPE_RELATIONS_H_
