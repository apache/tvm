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
 * \file src/ir/adt.cc
 * \brief ADT type definitions.
 */
#include <tvm/ir/adt.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/registry.h>

namespace tvm {

Constructor::Constructor(String name_hint, tvm::Array<Type> inputs, GlobalTypeVar belong_to) {
  ObjectPtr<ConstructorNode> n = make_object<ConstructorNode>();
  n->name_hint = std::move(name_hint);
  n->inputs = std::move(inputs);
  n->belong_to = std::move(belong_to);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ConstructorNode);

TVM_REGISTER_GLOBAL("ir.Constructor")
    .set_body_typed([](String name_hint, tvm::Array<Type> inputs, GlobalTypeVar belong_to) {
      return Constructor(name_hint, inputs, belong_to);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ConstructorNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ConstructorNode*>(ref.get());
      p->stream << "ConstructorNode(" << node->name_hint << ", " << node->inputs << ", "
                << node->belong_to << ")";
    });

TypeData::TypeData(GlobalTypeVar header, tvm::Array<TypeVar> type_vars,
                   tvm::Array<Constructor> constructors) {
  ObjectPtr<TypeDataNode> n = make_object<TypeDataNode>();
  n->header = std::move(header);
  n->type_vars = std::move(type_vars);
  n->constructors = std::move(constructors);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TypeDataNode);

TVM_REGISTER_GLOBAL("ir.TypeData")
    .set_body_typed([](GlobalTypeVar header, tvm::Array<TypeVar> type_vars,
                       tvm::Array<Constructor> constructors) {
      return TypeData(header, type_vars, constructors);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TypeDataNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TypeDataNode*>(ref.get());
      p->stream << "TypeDataNode(" << node->header << ", " << node->type_vars << ", "
                << node->constructors << ")";
    });

}  // namespace tvm
