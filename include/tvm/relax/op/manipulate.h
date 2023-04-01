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
#ifndef TVM_RELAX_OP_MANIPULATE_H_
#define TVM_RELAX_OP_MANIPULATE_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/index_map.h>

namespace tvm {
namespace relax {

// (TVM-TOOL) cc_op begin decl/manipulate/*
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param shape TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call broadcast_to(relax::Expr x, relax::Expr shape);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call concat(Array<relax::Expr> x, int64_t axis);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call expand_dims(relax::Expr x, Array<IntImm> axis);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param start_dim TODO(tvm-unity-team): add doc
 * \param end_dim TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call flatten(relax::Expr x, int64_t start_dim, int64_t end_dim);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param index_map TODO(tvm-unity-team): add doc
 * \param pad_value TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call layout_transform(relax::Expr x, tir::IndexMap index_map, Optional<FloatImm> pad_value);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param axes TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call permute_dims(relax::Expr x, Optional<Array<IntImm>> axes);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param repeats TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call repeat(relax::Expr x, Array<PrimExpr> repeats, Optional<IntImm> axis);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param shape TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call reshape(relax::Expr x, relax::Expr shape);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param indices_or_sections TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call split(relax::Expr x, ObjectRef indices_or_sections, int64_t axis);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call squeeze(relax::Expr x, Optional<Array<IntImm>> axis);
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param repeats TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call tile(relax::Expr x, relax::Expr repeats);
// (TVM-TOOL) cc_op end decl/manipulate/*

}  // namespace relax
}  // namespace tvm
#endif
