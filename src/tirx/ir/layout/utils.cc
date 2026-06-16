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

#include "utils.h"

namespace tvm {
namespace tirx {

Array<PrimExpr> SplitCoord(PrimExpr coord, const Array<PrimExpr>& shape) {
  Array<PrimExpr> result;
  for (int i = shape.size() - 1; i >= 0; --i) {
    if (i == 0) {
      result.push_back(coord);
    } else {
      result.push_back(floormod(coord, shape[i]));
      coord = floordiv(coord, shape[i]);
    }
  }
  return Array<PrimExpr>(result.rbegin(), result.rend());
}

PrimExpr FlattenCoord(const Array<PrimExpr>& coord, const Array<PrimExpr>& shape) {
  return std::accumulate(
      coord.begin(), coord.end(), PrimExpr(0),
      [&shape, i = 0](PrimExpr acc, const PrimExpr& c) mutable { return acc * shape[i++] + c; });
}

TileLayout IdentityTileLayout(const ffi::Array<PrimExpr>& shape) {
  if (shape.empty()) {
    // Degenerate identity: no shard dims.
    return TileLayout({}, {}, {});
  }
  PrimExpr extent = std::accumulate(shape.begin() + 1, shape.end(), shape[0],
                                    [](PrimExpr a, PrimExpr b) { return a * b; });
  return TileLayout({Iter(extent, 1, Axis::Get("m"))}, {}, {});
}

ffi::Map<ffi::String, PrimExpr> BuildSpanMap(const TileLayout& layout) {
  ffi::Map<ffi::String, PrimExpr> span_map;
  for (const auto& iter : layout->shard) {
    if (span_map.find(iter->axis->name) == span_map.end()) {
      span_map.Set(iter->axis->name, layout->GetSpan(iter->axis->name));
    }
  }
  return span_map;
}

std::vector<PrimExpr> GetDefaultStrides(const ffi::Array<PrimExpr>& data, PrimExpr initial_stride) {
  std::vector<PrimExpr> strides;
  if (data.empty()) return strides;
  size_t n = data.size();
  strides.resize(n);
  // Promote ``initial_stride`` (an IntImm constructed from `1`, defaults to
  // int32) to the dtype of the shape extents so the resulting strides
  // match what the tvmscript parser produces (``stride *= shape[i]`` in
  // Python preserves the shape's dtype). Otherwise int64-shaped buffers
  // get int32 strides and structurally differ from parser output.
  PrimExpr current_stride = initial_stride;
  if (const auto* imm = current_stride.as<IntImmNode>()) {
    current_stride = make_const(data[0].dtype(), imm->value);
  }
  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    strides[i] = current_stride;
    current_stride *= data[i];
  }
  return strides;
}

bool AxisMatchesFilter(const Axis& axis, const ffi::Optional<ffi::String>& axis_name) {
  return (!axis_name.has_value() && axis->IsMemoryAxis()) ||
         (axis_name.has_value() && axis->name == axis_name.value());
}

}  // namespace tirx
}  // namespace tvm
