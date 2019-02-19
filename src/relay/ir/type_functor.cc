/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_functor.cc
 * \brief Implementations of type functors.
 */
#include "type_functor.h"

namespace tvm {
namespace relay {

void TypeVisitor::VisitType_(const TypeVarNode* op) {
}

void TypeVisitor::VisitType_(const TensorTypeNode* op) {
}

void TypeVisitor::VisitType_(const IncompleteTypeNode* op) {
}

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

void TypeVisitor::VisitType_(const RefTypeNode* op) {
  this->VisitType(op->value);
}

void TypeVisitor::VisitType_(const TypeRelationNode* op) {
  for (const Type& t : op->args) {
    this->VisitType(t);
  }
}

void TypeVisitor::VisitType_(const GlobalTypeVarNode* op) {
}

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

// Type Mutator.
Array<Type> TypeMutator::MutateArray(Array<Type> arr) {
  // The array will do copy on write
  // If no changes are made, the original array will be returned.
  for (size_t i = 0; i < arr.size(); ++i) {
    Type ty = arr[i];
    Type new_ty = VisitType(ty);
    if (!ty.same_as(new_ty)) {
      arr.Set(i, new_ty);
    }
  }
  return arr;
}

Type TypeMutator::VisitType_(const TypeVarNode* op) {
  return GetRef<TypeVar>(op);
}

Type TypeMutator::VisitType_(const TensorTypeNode* op) {
  // TODO(tvm-team) recursively visit to replace Var
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const IncompleteTypeNode* op) {
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const FuncTypeNode* op) {
  bool changed = false;
  Array<TypeVar> type_params;
  for (auto type_param : op->type_params) {
    auto new_type_param = VisitType(type_param);
    changed = changed || !new_type_param.same_as(type_param);
    if (const TypeVarNode* tin = new_type_param.as<TypeVarNode>()) {
      type_params.push_back(GetRef<TypeVar>(tin));
    } else {
      LOG(FATAL) << new_type_param << std::endl;
    }
  }

  Array<TypeConstraint> type_constraints;
  for (auto type_cs : op->type_constraints) {
    auto new_type_cs = VisitType(type_cs);
    changed = changed || !new_type_cs.same_as(type_cs);
    if (const TypeConstraintNode* tin =
        new_type_cs.as_derived<TypeConstraintNode>()) {
      type_constraints.push_back(GetRef<TypeConstraint>(tin));
    } else {
      LOG(FATAL) << new_type_cs << std::endl;
    }
  }

  Array<Type> new_args = MutateArray(op->arg_types);
  changed = changed || !new_args.same_as(op->arg_types);

  Type new_ret_type = VisitType(op->ret_type);
  changed = changed || !new_ret_type.same_as(op->ret_type);

  if (!changed) return GetRef<Type>(op);
  return FuncTypeNode::make(new_args,
                            new_ret_type,
                            type_params,
                            type_constraints);
}

Type TypeMutator::VisitType_(const TupleTypeNode* op) {
  Array<Type> new_fields = MutateArray(op->fields);
  if (new_fields.same_as(op->fields)) {
    return GetRef<Type>(op);
  } else {
    return TupleTypeNode::make(new_fields);
  }
}

Type TypeMutator::VisitType_(const RefTypeNode* op) {
  return RefTypeNode::make(this->VisitType(op->value));
}

Type TypeMutator::VisitType_(const TypeRelationNode* type_rel) {
  Array<Type> new_args = MutateArray(type_rel->args);
  if (new_args.same_as(type_rel->args)) {
    return GetRef<Type>(type_rel);
  } else {
    return TypeRelationNode::make(type_rel->func,
                                  new_args,
                                  type_rel->num_inputs,
                                  type_rel->attrs);
  }
}

Type TypeMutator::VisitType_(const GlobalTypeVarNode* op) {
  return GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const TypeCallNode* op) {
  Type new_func = VisitType(op->func);
  Array<Type> new_args = MutateArray(op->args);
  if (new_args.same_as(op->args) && new_func.same_as(op->func)) {
    return GetRef<TypeCall>(op);
  } else {
    return TypeCallNode::make(new_func, new_args);
  }
}

Type TypeMutator::VisitType_(const TypeDataNode* op) {
  return GetRef<Type>(op);
}

// Implements bind.
class TypeBinder : public TypeMutator {
 public:
  explicit TypeBinder(const tvm::Map<TypeVar, Type>& args_map)
    : args_map_(args_map) {}

  Type VisitType_(const TypeVarNode* op) override {
    auto id = GetRef<TypeVar>(op);
    auto it = args_map_.find(id);
    if (it != args_map_.end()) {
      return (*it).second;
    } else {
      return id;
    }
  }

 private:
  const tvm::Map<TypeVar, Type>& args_map_;
};

Type Bind(const Type& type, const tvm::Map<TypeVar, Type>& args_map) {
  return TypeBinder(args_map).VisitType(type);
}

}  // namespace relay
}  // namespace tvm
