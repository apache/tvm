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

/*
 * Internal helpers for TileLayout implementations.
 * This header is private to the layout implementation files.
 */

#ifndef TVM_TIRX_IR_LAYOUT_TILE_INTERNAL_H_
#define TVM_TIRX_IR_LAYOUT_TILE_INTERNAL_H_

#include "utils.h"

namespace tvm {
namespace tirx {

// Group a tile layout's shard by a logical shape, returning the grouped layout and separators.
std::pair<TileLayout, std::vector<int64_t>> Group(TileLayout layout,
                                                  const ffi::Array<PrimExpr>& shape);

// Same as Group but returns std::nullopt instead of fatal-checking when the
// layout cannot be regrouped by ``shape``.
std::optional<std::pair<TileLayout, std::vector<int64_t>>> TryGroup(
    TileLayout layout, const ffi::Array<PrimExpr>& shape);

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_IR_LAYOUT_TILE_INTERNAL_H_
