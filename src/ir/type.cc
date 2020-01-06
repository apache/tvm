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
 * \file src/tvm/ir/type.cc
 * \brief Common type system AST nodes throughout the IR.
 */
#include <tvm/ir/type.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>

namespace tvm {

TypeVar TypeVarNode::make(std::string name, TypeKind kind) {
  ObjectPtr<TypeVarNode> n = make_object<TypeVarNode>();
  n->name_hint = std::move(name);
  n->kind = std::move(kind);
  return TypeVar(n);
}

TVM_REGISTER_NODE_TYPE(TypeVarNode);

TVM_REGISTER_GLOBAL("relay._make.TypeVar")
.set_body_typed([](std::string name, int kind) {
  return TypeVarNode::make(name, static_cast<TypeKind>(kind));
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<TypeVarNode>([](const ObjectRef& ref, NodePrinter* p) {
    auto* node = static_cast<const TypeVarNode*>(ref.get());
    p->stream << "TypeVar(" << node->name_hint << ", "
              << node->kind << ")";
});

GlobalTypeVar GlobalTypeVarNode::make(std::string name, TypeKind kind) {
  ObjectPtr<GlobalTypeVarNode> n = make_object<GlobalTypeVarNode>();
  n->name_hint = std::move(name);
  n->kind = std::move(kind);
  return GlobalTypeVar(n);
}

TVM_REGISTER_NODE_TYPE(GlobalTypeVarNode);

TVM_REGISTER_GLOBAL("relay._make.GlobalTypeVar")
.set_body_typed([](std::string name, int kind) {
  return GlobalTypeVarNode::make(name, static_cast<TypeKind>(kind));
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<GlobalTypeVarNode>([](const ObjectRef& ref, NodePrinter* p) {
    auto* node = static_cast<const GlobalTypeVarNode*>(ref.get());
    p->stream << "GlobalTypeVar(" << node->name_hint << ", "
              << node->kind << ")";
});

FuncType FuncTypeNode::make(tvm::Array<Type> arg_types,
                            Type ret_type,
                            tvm::Array<TypeVar> type_params,
                            tvm::Array<TypeConstraint> type_constraints) {
  ObjectPtr<FuncTypeNode> n = make_object<FuncTypeNode>();
  n->arg_types = std::move(arg_types);
  n->ret_type = std::move(ret_type);
  n->type_params = std::move(type_params);
  n->type_constraints = std::move(type_constraints);
  return FuncType(n);
}

TVM_REGISTER_NODE_TYPE(FuncTypeNode);

TVM_REGISTER_GLOBAL("relay._make.FuncType")
.set_body_typed(FuncTypeNode::make);

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<FuncTypeNode>([](const ObjectRef& ref, NodePrinter* p) {
  auto* node = static_cast<const FuncTypeNode*>(ref.get());
  p->stream << "FuncType(" << node->type_params << ", "
            << node->arg_types << ", " << node->ret_type << ", "
            << node->type_constraints << ")";
});

}  // namespace tvm
