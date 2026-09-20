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
#ifndef TVM_TIRX_BUFFER_REGION_H_
#define TVM_TIRX_BUFFER_REGION_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/tirx/buffer.h>

namespace tvm {
namespace tirx {

/*! \brief The type of a multi-dimensional buffer region expression. */
class BufferRegionTypeNode : public TypeNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BufferRegionTypeNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.BufferRegionType", BufferRegionTypeNode, TypeNode);
};

/*! \brief Managed reference to BufferRegionTypeNode. */
class BufferRegionType : public Type {
 public:
  TVM_DLL BufferRegionType();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BufferRegionType, Type, BufferRegionTypeNode);
};

/*! \brief Construct a region with buffer rank validation and BufferRegionType. */
TVM_DLL TensorRegion BufferRegion(BufferVar buffer, ffi::Array<Range> region, Span span = Span());
/*! \brief Select the entire buffer. */
TVM_DLL TensorRegion FullBufferRegion(BufferVar buffer);
/*! \brief Construct unit or vector-lane ranges from point indices. */
TVM_DLL TensorRegion BufferRegionFromPoint(BufferVar buffer, ffi::Array<PrimExpr> indices);

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_BUFFER_REGION_H_
