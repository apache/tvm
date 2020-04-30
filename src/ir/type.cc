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

PrimType::PrimType(runtime::DataType dtype) {
  ObjectPtr<PrimTypeNode> n = make_object<PrimTypeNode>();
  n->dtype = dtype;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PrimTypeNode);

TVM_REGISTER_GLOBAL("ir.PrimType")
.set_body_typed([](runtime::DataType dtype) {
  return PrimType(dtype);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<PrimTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const PrimTypeNode*>(ref.get());
    p->stream << node->dtype;
});


PointerType::PointerType(Type element_type) {
  ObjectPtr<PointerTypeNode> n = make_object<PointerTypeNode>();
  n->element_type = std::move(element_type);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PointerTypeNode);

TVM_REGISTER_GLOBAL("ir.PointerType")
.set_body_typed([](Type element_type) {
  return PointerType(element_type);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<PointerTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const PointerTypeNode*>(ref.get());
    p->Print(node->element_type);
    p->stream << '*';
});


TypeVar::TypeVar(String name, TypeKind kind) {
  ObjectPtr<TypeVarNode> n = make_object<TypeVarNode>();
  n->name_hint = std::move(name);
  n->kind = std::move(kind);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TypeVarNode);

TVM_REGISTER_GLOBAL("ir.TypeVar")
.set_body_typed([](String name, int kind) {
  return TypeVar(name, static_cast<TypeKind>(kind));
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<TypeVarNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const TypeVarNode*>(ref.get());
    p->stream << "TypeVar(" << node->name_hint << ", "
              << node->kind << ")";
});


GlobalTypeVar::GlobalTypeVar(std::string name, TypeKind kind) {
  ObjectPtr<GlobalTypeVarNode> n = make_object<GlobalTypeVarNode>();
  n->name_hint = std::move(name);
  n->kind = std::move(kind);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(GlobalTypeVarNode);

TVM_REGISTER_GLOBAL("ir.GlobalTypeVar")
.set_body_typed([](std::string name, int kind) {
  return GlobalTypeVar(name, static_cast<TypeKind>(kind));
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<GlobalTypeVarNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const GlobalTypeVarNode*>(ref.get());
    p->stream << "GlobalTypeVar(" << node->name_hint << ", "
              << node->kind << ")";
});

FuncType::FuncType(tvm::Array<Type> arg_types,
                   Type ret_type,
                   tvm::Array<TypeVar> type_params,
                   tvm::Array<TypeConstraint> type_constraints) {
  ObjectPtr<FuncTypeNode> n = make_object<FuncTypeNode>();
  n->arg_types = std::move(arg_types);
  n->ret_type = std::move(ret_type);
  n->type_params = std::move(type_params);
  n->type_constraints = std::move(type_constraints);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(FuncTypeNode);

TVM_REGISTER_GLOBAL("ir.FuncType")
.set_body_typed([](tvm::Array<Type> arg_types,
                   Type ret_type,
                   tvm::Array<TypeVar> type_params,
                   tvm::Array<TypeConstraint> type_constraints) {
  return FuncType(arg_types, ret_type, type_params, type_constraints);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<FuncTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const FuncTypeNode*>(ref.get());
  p->stream << "FuncType(" << node->type_params << ", "
            << node->arg_types << ", " << node->ret_type << ", "
            << node->type_constraints << ")";
});


TupleType::TupleType(Array<Type> fields) {
  ObjectPtr<TupleTypeNode> n = make_object<TupleTypeNode>();
  n->fields = std::move(fields);
  data_ = std::move(n);
}

TupleType TupleType::Empty() {
  return TupleType(Array<Type>());
}

TVM_REGISTER_NODE_TYPE(TupleTypeNode);

TVM_REGISTER_GLOBAL("ir.TupleType")
.set_body_typed([](Array<Type> fields) {
  return TupleType(fields);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<TupleTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const TupleTypeNode*>(ref.get());
  p->stream << "TupleTypeNode(" << node->fields << ")";
});


IncompleteType::IncompleteType(TypeKind kind) {
  auto n = make_object<IncompleteTypeNode>();
  n->kind = std::move(kind);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(IncompleteTypeNode);

TVM_REGISTER_GLOBAL("ir.IncompleteType")
.set_body_typed([](int kind) {
    return IncompleteType(static_cast<TypeKind>(kind));
  });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<IncompleteTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const IncompleteTypeNode*>(ref.get());
    p->stream << "IncompleteTypeNode(" << node->kind << ", " << node << ")";
  });


RelayRefType::RelayRefType(Type value) {
  ObjectPtr<RelayRefTypeNode> n = make_object<RelayRefTypeNode>();
  n->value = std::move(value);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("ir.RelayRefType")
.set_body_typed([](Type value) {
  return RelayRefType(value);
});

TVM_REGISTER_NODE_TYPE(RelayRefTypeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<RelayRefTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const RelayRefTypeNode*>(ref.get());
  p->stream << "RelayRefTypeNode(" << node->value << ")";
});

}  // namespace tvm
