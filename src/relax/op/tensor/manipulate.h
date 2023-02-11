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
 * \file manipulate.h
 * \brief The functions to make Relax tensor manipulation operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_MANIPULATE_H_
#define TVM_RELAX_OP_TENSOR_MANIPULATE_H_

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Reshape the input array, supporting `-1` inference in the new
 * shape when the new shape is given as an Array of PrimExpr.
 * \param x The input data to the operator.
 * \param shape The new shape. Should be compatible with the original shape.
 * It is required to be either an Array of PrimExpr, or a Shape in Relax
 * \return The reshaped result.
 */
Expr reshape(Expr x, ObjectRef shape);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_MANIPULATE_H_
