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

/*!
 * \file src/relax/ir/type.cc
 * \brief Relax type system.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/type.h>

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  ShapeTypeNode::RegisterReflection();
  TensorTypeNode::RegisterReflection();
  ObjectTypeNode::RegisterReflection();
  PackedFuncTypeNode::RegisterReflection();
}

ShapeType::ShapeType(int ndim, Span span) {
  ObjectPtr<ShapeTypeNode> n = ffi::make_object<ShapeTypeNode>();
  n->ndim = ndim;
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ShapeType",
                        [](int ndim, Span span) { return ShapeType(ndim, span); });
}

ObjectType::ObjectType(Span span) {
  ObjectPtr<ObjectTypeNode> n = ffi::make_object<ObjectTypeNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ObjectType", [](Span span) { return ObjectType(span); });
}

TensorType::TensorType(int ndim, DataType dtype, Span span) {
  ObjectPtr<TensorTypeNode> n = ffi::make_object<TensorTypeNode>();
  n->ndim = std::move(ndim);
  n->dtype = std::move(dtype);
  n->span = span;
  data_ = std::move(n);
}

TensorType TensorType::CreateUnknownNDim(DataType dtype, Span span) {
  ObjectPtr<TensorTypeNode> n = ffi::make_object<TensorTypeNode>();
  n->ndim = -1;
  n->dtype = std::move(dtype);
  n->span = std::move(span);
  return TensorType(std::move(n));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.TensorType", [](int ndim, DataType dtype, Span span) {
    return TensorType(ndim, dtype, span);
  });
}

PackedFuncType::PackedFuncType(Span span) {
  ObjectPtr<PackedFuncTypeNode> n = ffi::make_object<PackedFuncTypeNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.PackedFuncType", [](Span span) { return PackedFuncType(span); });
}

}  // namespace relax
}  // namespace tvm
