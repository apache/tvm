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
/*! \file tvm/s_tir/layout.h
 * \brief Strided tensor adapters for S-TIR's regular-memory contracts.
 */
#ifndef TVM_S_TIR_LAYOUT_H_
#define TVM_S_TIR_LAYOUT_H_

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/type.h>

namespace tvm::s_tir {

inline ffi::Optional<tirx::Layout> WithStrides(const ffi::Optional<tirx::Layout>& layout,
                                               const ffi::Array<PrimExpr>& shape,
                                               ffi::Array<PrimExpr> strides) {
  if (!layout.has_value() && strides.empty()) return std::nullopt;
  auto tile = layout.has_value() ? layout.value().as_or_throw<tirx::TileLayout>()
                                 : tirx::TileLayout({}, {}, {});

  if (strides.empty()) {
    PrimExpr stride = IntImm(tirx::DefaultIndexType(), 1);
    strides = ffi::Array<PrimExpr>(shape.size(), stride);
    for (size_t i = shape.size(); i > 0; --i) {
      strides.Set(i - 1, stride);
      stride = stride * shape[i - 1];
    }
  }
  TVM_FFI_CHECK(shape.size() == strides.size(), ValueError) << "Stride rank must match shape";
  ffi::Array<tirx::Iter> shard;
  for (size_t i = 0; i < shape.size(); ++i) {
    shard.push_back(tirx::Iter(shape[i], strides[i], tirx::Axis::Get("m")));
  }
  return tirx::TileLayout(std::move(shard), tile->replica, tile->offset);
}

}  // namespace tvm::s_tir
#endif  // TVM_S_TIR_LAYOUT_H_
