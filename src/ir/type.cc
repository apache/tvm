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
 * \file src/ir/type.cc
 * \brief Common type system AST nodes throughout the IR.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>

#include <cstdint>
#include <unordered_map>

namespace tvm {

namespace {

DLDataType ScalableVectorDType(DLDataTypeCode code, int bits, int lanes) {
  TVM_FFI_ICHECK_GT(lanes, 1) << "Invalid value for vscale factor " << lanes;
  TVM_FFI_ICHECK_LT(lanes, 32768);
  return DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits),
                    static_cast<uint16_t>(-lanes)};
}

uint32_t PackDataTypeKey(DLDataType dtype) {
  return (static_cast<uint32_t>(dtype.code) << 24) | (static_cast<uint32_t>(dtype.bits) << 16) |
         static_cast<uint32_t>(dtype.lanes);
}

int64_t PrimTypeAnyHash(const ffi::Any& src) {
  return static_cast<int64_t>(PackDataTypeKey(src.cast<PrimType>()->dtype));
}

bool PrimTypeAnyEqual(const ffi::Any& lhs, const ffi::Any& rhs) {
  return lhs.cast<PrimType>()->dtype == rhs.cast<PrimType>()->dtype;
}

ffi::ObjectPtr<PrimTypeNode> GetCachedPrimTypeNode(DLDataType dtype) {
  thread_local std::unordered_map<uint32_t, ffi::ObjectPtr<PrimTypeNode>> cache;
  uint32_t key = PackDataTypeKey(dtype);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  ffi::ObjectPtr<PrimTypeNode> node = ffi::make_object<PrimTypeNode>();
  node->dtype = dtype;
  return cache.emplace(key, std::move(node)).first->second;
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TypeNode::RegisterReflection();
  PrimTypeNode::RegisterReflection();
  refl::TypeAttrDef<PrimTypeNode>()
      .attr(refl::type_attr::kAnyHash, reinterpret_cast<void*>(&PrimTypeAnyHash))
      .attr(refl::type_attr::kAnyEqual, reinterpret_cast<void*>(&PrimTypeAnyEqual));
  PointerTypeNode::RegisterReflection();
  TupleTypeNode::RegisterReflection();
  FuncTypeNode::RegisterReflection();
  TensorMapTypeNode::RegisterReflection();
}

PrimType::PrimType(DLDataType dtype) { data_ = GetCachedPrimTypeNode(dtype); }

PrimType::PrimType(DLDataTypeCode code, int bits, int lanes)
    : PrimType(DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits),
                          static_cast<uint16_t>(lanes)}) {}

PrimType PrimType::Int(int bits, int lanes) {
  if (lanes == 1) {
    if (bits == 32) {
      thread_local PrimType i32_ty(DLDataType{kDLInt, 32, 1});
      return i32_ty;
    }
    if (bits == 64) {
      thread_local PrimType i64_ty(DLDataType{kDLInt, 64, 1});
      return i64_ty;
    }
  }
  return PrimType(DLDataType{kDLInt, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::UInt(int bits, int lanes) {
  return PrimType(DLDataType{kDLUInt, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::Float(int bits, int lanes) {
  if (bits == 32 && lanes == 1) {
    thread_local PrimType f32_ty(DLDataType{kDLFloat, 32, 1});
    return f32_ty;
  }
  return PrimType(DLDataType{kDLFloat, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::BFloat(int bits, int lanes) {
  return PrimType(DLDataType{kDLBfloat, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::Bool(int lanes) {
  if (lanes == 1) {
    thread_local PrimType bool_ty(DLDataType{kDLBool, 8, 1});
    return bool_ty;
  }
  return PrimType(DLDataType{kDLBool, 8, static_cast<uint16_t>(lanes)});
}

PrimType PrimType::Handle(int bits, int lanes) {
  return PrimType(
      DLDataType{kDLOpaqueHandle, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::Void() { return PrimType(DLDataType{kDLOpaqueHandle, 0, 0}); }

PrimType PrimType::ScalableVector(DLDataTypeCode code, int bits, int lanes) {
  return PrimType(ScalableVectorDType(code, bits, lanes));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.PrimType", [](DLDataType dtype) { return PrimType(dtype); });
}

PointerType::PointerType(Type element_type, ffi::String storage_scope) {
  ffi::ObjectPtr<PointerTypeNode> n = ffi::make_object<PointerTypeNode>();
  if (storage_scope.empty()) {
    n->storage_scope = "global";
  } else {
    n->storage_scope = std::move(storage_scope);
  }
  n->element_type = std::move(element_type);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.PointerType", [](Type element_type, ffi::String storage_scope = "") {
    return PointerType(element_type, storage_scope);
  });
}

FuncType::FuncType(tvm::ffi::Array<Type> arg_types, Type ret_type, Span span) {
  ffi::ObjectPtr<FuncTypeNode> n = ffi::make_object<FuncTypeNode>();
  n->arg_types = std::move(arg_types);
  n->ret_type = std::move(ret_type);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.FuncType", [](tvm::ffi::Array<Type> arg_types, Type ret_type) {
    return FuncType(arg_types, ret_type);
  });
}

TupleType::TupleType(ffi::Array<Type> fields, Span span) {
  ffi::ObjectPtr<TupleTypeNode> n = ffi::make_object<TupleTypeNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

TupleType TupleType::Empty() { return TupleType(ffi::Array<Type>()); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ir.TupleType",
           [](ffi::Array<Type> fields, Span span) { return TupleType(fields, span); })
      .def("ir.TensorMapType", [](Span span) { return TensorMapType(span); });
}

TensorMapType::TensorMapType(Span span) {
  ffi::ObjectPtr<TensorMapTypeNode> n = ffi::make_object<TensorMapTypeNode>();
  n->span = std::move(span);
  data_ = std::move(n);
}

}  // namespace tvm
