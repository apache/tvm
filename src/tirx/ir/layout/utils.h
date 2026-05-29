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

#ifndef TVM_TIRX_IR_LAYOUT_UTILS_H_
#define TVM_TIRX_IR_LAYOUT_UTILS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op.h>

#include <numeric>
#include <vector>

#include "../../../ir/attr_registry.h"

namespace tvm {
namespace tirx {

using ffi::StructuralEqual;
using ffi::StructuralHash;

/*!
 * \brief Split the coordinate into multiple parts
 * \param coord The coordinate to split
 * \param shape The shape of the tensor
 * \return The split coordinates
 */
Array<PrimExpr> SplitCoord(PrimExpr coord, const Array<PrimExpr>& shape);

/*!
 * \brief Flatten the split coordinates
 * \param coord The split coordinates
 * \param shape The shape of the tensor
 * \return The flattened coordinate
 */
PrimExpr FlattenCoord(const Array<PrimExpr>& coord, const Array<PrimExpr>& shape);

/*!
 * \brief Create a TileLayout that maps the given logical shape to itself on the memory axis.
 *        This is effectively an identity layout over axis "m" with unit stride.
 * \param shape Logical shape to map.
 * \return Identity TileLayout over the concatenated extent of `shape`.
 */
TileLayout IdentityTileLayout(const ffi::Array<PrimExpr>& shape);

/*!
 * \brief Build a map from axis name to span for the provided layout's shard axes.
 *        If an axis appears multiple times, the first occurrence defines the span value.
 * \param layout The layout whose shard axes will be scanned.
 * \return A map from axis name to span expression.
 */
ffi::Map<ffi::String, PrimExpr> BuildSpanMap(const TileLayout& layout);

/*!
 * \brief Compute default contiguous strides for a list of extents.
 *        The last dimension has `initial_stride`, and strides accumulate outward.
 * \param data The extents per dimension.
 * \param initial_stride The initial innermost stride, defaults to 1.
 * \return A vector of strides, same length as `data`.
 */
std::vector<PrimExpr> GetDefaultStrides(const ffi::Array<PrimExpr>& data,
                                        PrimExpr initial_stride = PrimExpr(1));

/*!
 * \brief Test whether an axis matches the optional axis_name filter used by size/span queries.
 *        When `axis_name` is not provided, memory axes match; when provided, the name must match.
 */
bool AxisMatchesFilter(const Axis& axis, const ffi::Optional<ffi::String>& axis_name);

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_IR_LAYOUT_UTILS_H_
