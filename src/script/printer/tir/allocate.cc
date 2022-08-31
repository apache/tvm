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
#include "./allocate.h"

#include "./buffer.h"

namespace tvm {
namespace script {
namespace printer {

AllocateUsage FindAllocateUsage(const TracedObject<tir::Var>& allocated_ptr, DataType dtype,
                             const Array<PrimExpr>& extents, const TracedObject<tir::Stmt>& body) {
  std::vector<TracedObject<tir::Buffer>> aliasing_buffers =
      FindAliasingBuffers(allocated_ptr.Get(), body);

  auto is_exact_match = [dtype, extents](const tir::Buffer& buf) {
    if (buf->dtype != dtype) return false;
    if (buf->shape.size() != extents.size()) return false;

    arith::Analyzer analyzer;
    for (size_t i = 0; i < buf->shape.size(); i++) {
      if (!analyzer.CanProveEqual(buf->shape[i], extents[i])) {
        return false;
      }
    }
    return true;
  };

  // If the buffer allocated via T.allocate is an exact match to the
  // usage of the buffer later on, then that buffer is the return
  // value of T.allocate, and no T.buffer_decl statement is needed.
  AllocateUsage ret = {TracedOptional<tir::Buffer>(NullOpt, ObjectPath::Root()), {}};
  for (const auto& buf : aliasing_buffers) {
    if (!ret.alloc_buffer.defined() && is_exact_match(buf.Get())) {
      ret.alloc_buffer = buf;
    } else {
      ret.aliasing_buffers.push_back(buf);
    }
  }
  return ret;
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
