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
#include <tvm/ir/type.h>
#include <tvm/runtime/registry.h>
namespace tvm {

PrimType::PrimType(runtime::DataType dtype, Span span) {
  ObjectPtr<PrimTypeNode> n = make_object<PrimTypeNode>();
  n->dtype = dtype;
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PrimTypeNode);

TVM_REGISTER_GLOBAL("ir.PrimType").set_body_typed([](runtime::DataType dtype) {
  return PrimType(dtype);
});

PointerType::PointerType(Type element_type, String storage_scope) {
  ObjectPtr<PointerTypeNode> n = make_object<PointerTypeNode>();
  n->element_type = std::move(element_type);
  n->storage_scope = std::move(storage_scope);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PointerTypeNode);

TVM_REGISTER_GLOBAL("ir.PointerType")
    .set_body_typed([](Type element_type, String storage_scope = "") {
      return PointerType(element_type, storage_scope);
    });

TypeVar::TypeVar(String name, TypeKind kind, Span span) {
  ObjectPtr<TypeVarNode> n = make_object<TypeVarNode>();
  n->name_hint = std::move(name);
  n->kind = std::move(kind);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TypeVarNode);

TVM_REGISTER_GLOBAL("ir.TypeVar").set_body_typed([](String name, int kind) {
  return TypeVar(name, static_cast<TypeKind>(kind));
});

GlobalTypeVar::GlobalTypeVar(String name, TypeKind kind, Span span) {
  ObjectPtr<GlobalTypeVarNode> n = make_object<GlobalTypeVarNode>();
  n->name_hint = std::move(name);
  n->kind = std::move(kind);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(GlobalTypeVarNode);

TVM_REGISTER_GLOBAL("ir.GlobalTypeVar").set_body_typed([](String name, int kind) {
  return GlobalTypeVar(name, static_cast<TypeKind>(kind));
});

FuncType::FuncType(tvm::Array<Type> arg_types, Type ret_type, tvm::Array<TypeVar> type_params,
                   tvm::Array<TypeConstraint> type_constraints, Span span) {
  ObjectPtr<FuncTypeNode> n = make_object<FuncTypeNode>();
  n->arg_types = std::move(arg_types);
  n->ret_type = std::move(ret_type);
  n->type_params = std::move(type_params);
  n->type_constraints = std::move(type_constraints);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(FuncTypeNode);

TVM_REGISTER_GLOBAL("ir.FuncType")
    .set_body_typed([](tvm::Array<Type> arg_types, Type ret_type, tvm::Array<TypeVar> type_params,
                       tvm::Array<TypeConstraint> type_constraints) {
      return FuncType(arg_types, ret_type, type_params, type_constraints);
    });

TupleType::TupleType(Array<Type> fields, Span span) {
  ObjectPtr<TupleTypeNode> n = make_object<TupleTypeNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

TupleType TupleType::Empty() { return TupleType(Array<Type>()); }

TVM_REGISTER_NODE_TYPE(TupleTypeNode);

TVM_REGISTER_GLOBAL("ir.TupleType").set_body_typed([](Array<Type> fields) {
  return TupleType(fields);
});

IncompleteType::IncompleteType(TypeKind kind, Span span) {
  auto n = make_object<IncompleteTypeNode>();
  n->kind = std::move(kind);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(IncompleteTypeNode);

TVM_REGISTER_GLOBAL("ir.IncompleteType").set_body_typed([](int kind) {
  return IncompleteType(static_cast<TypeKind>(kind));
});

RelayRefType::RelayRefType(Type value, Span span) {
  ObjectPtr<RelayRefTypeNode> n = make_object<RelayRefTypeNode>();
  n->value = std::move(value);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("ir.RelayRefType").set_body_typed([](Type value) {
  return RelayRefType(value);
});

TVM_REGISTER_NODE_TYPE(RelayRefTypeNode);

}  // namespace tvm
