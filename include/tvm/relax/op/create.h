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
#ifndef TVM_RELAX_OP_CREATE_H_
#define TVM_RELAX_OP_CREATE_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>

namespace tvm {
namespace relax {

// (TVM-TOOL) cc_op begin decl/create/*
/*!
 * TBD
 * \param shape The shape of the output tensor.
 * \param fill_value The value to fill the output tensor with.
 * \param dtype The data type of the output tensor.
 * \return The output tensor.
 */
relax::Call full(relax::Expr shape, ObjectRef fill_value, runtime::DataType dtype);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param fill_value TODO(tvm-unity-team): add doc
 * \param dtype TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call full_like(relax::Expr x, ObjectRef fill_value, runtime::DataType dtype);
/*!
 * TBD
 * \param shape TODO(tvm-unity-team): add doc
 * \param dtype TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call ones(relax::Expr shape, runtime::DataType dtype);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param dtype TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call ones_like(relax::Expr x, runtime::DataType dtype);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param k TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call tril(relax::Expr x, PrimExpr k);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param k TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call triu(relax::Expr x, PrimExpr k);
/*!
 * TBD
 * \param shape TODO(tvm-unity-team): add doc
 * \param dtype TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call zeros(relax::Expr shape, runtime::DataType dtype);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param dtype TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call zeros_like(relax::Expr x, runtime::DataType dtype);
// (TVM-TOOL) cc_op end decl/create/*

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_CREATE_H_
