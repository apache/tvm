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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/type.h>

namespace tvm {
namespace relax {

PackedFuncType::PackedFuncType(Span span) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<PackedFuncTypeNode> n = ffi::make_object<PackedFuncTypeNode>();
  n->span = span;
  data_ = std::move(n);
}

namespace {

TVMFFIAny PackedFuncTypeVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny PackedFuncTypeMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny PackedFuncTypeMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  PackedFuncTypeNode::RegisterReflection();
  refl::TypeAttrDef<PackedFuncTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&PackedFuncTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&PackedFuncTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&PackedFuncTypeMaybeInplaceMutate));

  refl::GlobalDef().def("relax.PackedFuncType", [](Span span) { return PackedFuncType(span); });
}

}  // namespace relax
}  // namespace tvm
