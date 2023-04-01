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
#ifndef TVM_RELAX_OP_STATISTICAL_H_
#define TVM_RELAX_OP_STATISTICAL_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>

namespace tvm {
namespace relax {

// (TVM-TOOL) cc_op begin decl/statistical/*
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \param keepdims TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call max(relax::Expr x, Array<IntImm> axis, bool keepdims);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \param keepdims TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call mean(relax::Expr x, Array<IntImm> axis, bool keepdims);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \param keepdims TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call min(relax::Expr x, Array<IntImm> axis, bool keepdims);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \param keepdims TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call prod(relax::Expr x, Array<IntImm> axis, bool keepdims);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \param keepdims TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call std(relax::Expr x, Array<IntImm> axis, bool keepdims);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \param keepdims TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call sum(relax::Expr x, Array<IntImm> axis, bool keepdims);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \param keepdims TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call variance(relax::Expr x, Array<IntImm> axis, bool keepdims);
// (TVM-TOOL) cc_op end decl/statistical/*

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_STATISTICAL_H_
