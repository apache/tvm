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
 * \file src/ir/type_relation.cc
 * \brief Type relation
 */
#include <tvm/ir/type.h>
#include <tvm/ir/type_relation.h>
#include <tvm/runtime/registry.h>
namespace tvm {

TypeCall::TypeCall(Type func, tvm::Array<Type> args) {
  ObjectPtr<TypeCallNode> n = make_object<TypeCallNode>();
  n->func = std::move(func);
  n->args = std::move(args);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TypeCallNode);

TVM_REGISTER_GLOBAL("ir.TypeCall").set_body_typed([](Type func, Array<Type> type) {
  return TypeCall(func, type);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TypeCallNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TypeCallNode*>(ref.get());
      p->stream << "TypeCallNode(" << node->func << ", " << node->args << ")";
    });

TypeRelation::TypeRelation(TypeRelationFn func, Array<Type> args, int num_inputs, Attrs attrs) {
  ObjectPtr<TypeRelationNode> n = make_object<TypeRelationNode>();
  n->func = std::move(func);
  n->args = std::move(args);
  n->num_inputs = num_inputs;
  n->attrs = std::move(attrs);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TypeRelationNode);

TVM_REGISTER_GLOBAL("ir.TypeRelation")
    .set_body_typed([](TypeRelationFn func, Array<Type> args, int num_inputs, Attrs attrs) {
      return TypeRelation(func, args, num_inputs, attrs);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TypeRelationNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TypeRelationNode*>(ref.get());
      p->stream << "TypeRelationNode(" << node->func->name << ", " << node->args << ")";
    });
}  // namespace tvm
