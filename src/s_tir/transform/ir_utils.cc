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

#include "ir_utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace s_tir {
using namespace tirx;
using namespace tvm::prim;

ffi::Array<PrimExpr> ConvertIndices(const MatchBufferRegion& match_buffer,
                                    const ffi::Array<PrimExpr>& indices) {
  const BufferVar& target = match_buffer->buffer;
  const BufferRegion& source = match_buffer->source;
  TVM_FFI_ICHECK_EQ(indices.size(), target->shape.size());

  arith::Analyzer analyzer;
  ffi::Array<PrimExpr> result;
  result.reserve(source->region.size());
  size_t offset = source->region.size() - indices.size();
  for (size_t i = 0; i < offset; ++i) {
    const Range& range = source->region[i];
    TVM_FFI_ICHECK(analyzer->CanProve(range->extent == 1));
    result.push_back(range->min);
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    const Range& range = source->region[i + offset];
    const PrimExpr& index = indices[i];
    result.push_back(range->min + index);
  }
  return result;
}

Region ConvertRegion(const MatchBufferRegion& match_buffer, const Region& region) {
  const BufferVar& target = match_buffer->buffer;
  const BufferRegion& source = match_buffer->source;
  TVM_FFI_ICHECK_EQ(region.size(), target->shape.size());

  arith::Analyzer analyzer;
  Region result;
  result.reserve(source->region.size());
  size_t offset = source->region.size() - region.size();
  for (size_t i = 0; i < offset; ++i) {
    const Range& source_range = source->region[i];
    TVM_FFI_ICHECK(analyzer->CanProve(source_range->extent == 1));
    result.push_back(Range::FromMinExtent(source_range->min, 1));
  }
  for (size_t i = 0; i < region.size(); ++i) {
    const Range& source_range = source->region[i + offset];
    const Range& target_range = region[i];
    result.push_back(
        Range::FromMinExtent(source_range->min + target_range->min, target_range->extent));
  }
  return result;
}

}  // namespace s_tir
}  // namespace tvm
