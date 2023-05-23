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
 * \file type_functor.cc
 * \brief Implementations of type functors.
 */
#include <tvm/ir/type_functor.h>

#include <utility>

namespace tvm {

void TypeVisitor::VisitType_(const TypeVarNode* op) {}

void TypeVisitor::VisitType_(const TensorTypeNode* op) {}

void TypeVisitor::VisitType_(const IncompleteTypeNode* op) {}

void TypeVisitor::VisitType_(const FuncTypeNode* op) {
  for (auto type_param : op->type_params) {
    this->VisitType(type_param);
  }

  for (auto type_cs : op->type_constraints) {
    this->VisitType(type_cs);
  }

  for (auto arg_type : op->arg_types) {
    this->VisitType(arg_type);
  }
  this->VisitType(op->ret_type);
}

void TypeVisitor::VisitType_(const TupleTypeNode* op) {
  for (const Type& t : op->fields) {
    this->VisitType(t);
  }
}

void TypeVisitor::VisitType_(const RelayRefTypeNode* op) { this->VisitType(op->value); }

void TypeVisitor::VisitType_(const TypeRelationNode* op) {
  for (const Type& t : op->args) {
    this->VisitType(t);
  }
}

void TypeVisitor::VisitType_(const GlobalTypeVarNode* op) {}

void TypeVisitor::VisitType_(const TypeCallNode* op) {
  this->VisitType(op->func);
  for (const Type& t : op->args) {
    this->VisitType(t);
  }
}

void TypeVisitor::VisitType_(const TypeDataNode* op) {
  this->VisitType(op->header);
  for (const auto& v : op->type_vars) {
    this->VisitType(v);
  }

  for (const auto& c : op->constructors) {
    this->VisitType(c->belong_to);
    for (const auto& t : c->inputs) {
      this->VisitType(t);
    }
  }
}

void TypeVisitor::VisitType_(const PrimTypeNode* op) {}

void TypeVisitor::VisitType_(const PointerTypeNode* op) { this->VisitType(op->element_type); }

Type TypeMutator::VisitType(const Type& t) {
  return t.defined() ? TypeFunctor<Type(const Type&)>::VisitType(t) : t;
}

// Type Mutator.
Array<Type> TypeMutator::MutateArray(Array<Type> arr) {
  // The array will do copy on write
  // If no changes are made, the original array will be returned.
  return arr.Map([this](const Type& ty) { return VisitType(ty); });
}

Type TypeMutator::VisitType_(const TypeVarNode* op) { return GetRef<TypeVar>(op); }

Type TypeMutator::VisitType_(const TensorTypeNode* op) {
  // TODO(tvm-team) recursively visit to replace Var
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const IncompleteTypeNode* op) { return GetRef<Type>(op); }

Type TypeMutator::VisitType_(const FuncTypeNode* op) {
  bool changed = false;
  Array<TypeVar> type_params;
  for (auto type_param : op->type_params) {
    auto new_type_param = VisitType(type_param);
    changed = changed || !new_type_param.same_as(type_param);
    if (auto tin = new_type_param.as<TypeVar>()) {
      type_params.push_back(tin.value());
    } else {
      LOG(FATAL) << new_type_param;
    }
  }

  Array<TypeConstraint> type_constraints;
  for (auto type_cs : op->type_constraints) {
    auto new_type_cs = VisitType(type_cs);
    changed = changed || !new_type_cs.same_as(type_cs);
    if (auto tin = new_type_cs.as<TypeConstraint>()) {
      type_constraints.push_back(tin.value());
    } else {
      LOG(FATAL) << new_type_cs;
    }
  }

  Array<Type> new_args = MutateArray(op->arg_types);
  changed = changed || !new_args.same_as(op->arg_types);

  Type new_ret_type = VisitType(op->ret_type);
  changed = changed || !new_ret_type.same_as(op->ret_type);

  if (!changed) return GetRef<Type>(op);
  return FuncType(new_args, new_ret_type, type_params, type_constraints);
}

Type TypeMutator::VisitType_(const TupleTypeNode* op) {
  Array<Type> new_fields = MutateArray(op->fields);
  if (new_fields.same_as(op->fields)) {
    return GetRef<Type>(op);
  } else {
    return TupleType(new_fields);
  }
}

Type TypeMutator::VisitType_(const RelayRefTypeNode* op) {
  return RelayRefType(this->VisitType(op->value));
}

Type TypeMutator::VisitType_(const TypeRelationNode* type_rel) {
  Array<Type> new_args = MutateArray(type_rel->args);
  if (new_args.same_as(type_rel->args)) {
    return GetRef<Type>(type_rel);
  } else {
    return TypeRelation(type_rel->func, new_args, type_rel->num_inputs, type_rel->attrs);
  }
}

Type TypeMutator::VisitType_(const GlobalTypeVarNode* op) { return GetRef<Type>(op); }

Type TypeMutator::VisitType_(const TypeCallNode* op) {
  Type new_func = VisitType(op->func);
  Array<Type> new_args = MutateArray(op->args);
  if (new_args.same_as(op->args) && new_func.same_as(op->func)) {
    return GetRef<TypeCall>(op);
  } else {
    return TypeCall(new_func, new_args);
  }
}

Type TypeMutator::VisitType_(const TypeDataNode* op) { return GetRef<Type>(op); }

Type TypeMutator::VisitType_(const PrimTypeNode* op) { return GetRef<Type>(op); }

Type TypeMutator::VisitType_(const PointerTypeNode* op) {
  Type element_type = VisitType(op->element_type);

  if (element_type.same_as(op->element_type)) {
    return GetRef<Type>(op);
  } else {
    return PointerType(element_type, op->storage_scope);
  }
}

// Implements bind.
class TypeBinder : public TypeMutator {
 public:
  explicit TypeBinder(const tvm::Map<TypeVar, Type>& args_map) : args_map_(args_map) {}

  Type VisitType_(const TypeVarNode* op) override {
    auto id = GetRef<TypeVar>(op);
    auto it = args_map_.find(id);
    if (it != args_map_.end()) {
      return (*it).second;
    } else {
      return std::move(id);
    }
  }

 private:
  const tvm::Map<TypeVar, Type>& args_map_;
};

Type Bind(const Type& type, const tvm::Map<TypeVar, Type>& args_map) {
  return TypeBinder(args_map).VisitType(type);
}

}  // namespace tvm
