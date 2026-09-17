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
 * \file tirx/ir/type.cc
 * \brief Types specific to TIRX.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/type.h>

namespace tvm::tirx {
namespace {

TVMFFIAny TensorMapTypeVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny TensorMapTypeMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny TensorMapTypeMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

}  // namespace

TensorMapType::TensorMapType(Span span) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TensorMapTypeNode> n = ffi::make_object<TensorMapTypeNode>();
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorMapTypeNode::RegisterReflection();
  refl::TypeAttrDef<TensorMapTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&TensorMapTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&TensorMapTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TensorMapTypeMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.TensorMapType", [](Span span) { return TensorMapType(span); });
}

}  // namespace tvm::tirx
