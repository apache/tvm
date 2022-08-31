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
#ifndef TVM_SCRIPT_PRINTER_TIR_ALLOCATE_H_
#define TVM_SCRIPT_PRINTER_TIR_ALLOCATE_H_

#include <tvm/arith/analyzer.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>

#include <vector>

namespace tvm {
namespace script {
namespace printer {

struct AllocateUsage {
  TracedOptional<tir::Buffer> alloc_buffer;
  std::vector<TracedObject<tir::Buffer>> aliasing_buffers;
};

AllocateUsage FindAllocateUsage(const TracedObject<tir::Var>& allocated_ptr, DataType dtype,
                                const Array<PrimExpr>& extents,
                                const TracedObject<tir::Stmt>& body);

template <typename AllocRef>  // Template for Allocate and AllocateConst
AllocateUsage FindAllocateUsage(const TracedObject<AllocRef>& traced_stmt) {
  auto body = traced_stmt.GetAttr(&AllocRef::ContainerType::body);
  auto buffer_var = traced_stmt.GetAttr(&AllocRef::ContainerType::buffer_var);
  const AllocRef& stmt = traced_stmt.Get();

  return FindAllocateUsage(buffer_var, stmt->dtype, stmt->extents, body);
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TIR_ALLOCATE_H_
