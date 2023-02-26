/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
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
/*!
 * \brief The identity type relation, all the types are equal.
 *
 * \param types The input and output types to the relation.
 * \param num_inputs The number of input arguments.
 * \param attrs The attributes
 * \param reporter The reporter.
 * \return true whether relation has been resolved.
 */
bool IdentityRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
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
bool BroadcastRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter);

/*!
 * \brief Determine the broadcasted shape from two input shapes
 * \param t1 One of two Tensortype whose shapes are broadcasted
 * \param t2 One of two Tensortype whose shapes are broadcasted
 * \param output_dtype dtype of the output TensorType
 * \return A TensorType whose shape is broadcasted from two input TensorType.
 */
TensorType ConcreteBroadcast(const TensorType& t1, const TensorType& t2, DataType output_dtype);

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
bool BroadcastCompRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter);

bool IdentityCompRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter);

Array<IndexExpr> RankShape(const Array<IndexExpr>& shape);

/*!
 * \brief The shape of type relation.
 *
 * \param types The input and output types to the relation.
 * \param num_inputs The number of input arguments.
 * \param attrs The attributes
 * \param reporter The reporter.
 * \return true whether relation has been resolved.
 */
bool ShapeOfRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_TYPE_RELATIONS_H_
