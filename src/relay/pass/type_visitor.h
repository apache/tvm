/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_visitor.h
 * \brief A wrapper around TypeFunctor for common use cases.
 */
#ifndef TVM_RELAY_PASS_TYPE_VISITOR_H_
#define TVM_RELAY_PASS_TYPE_VISITOR_H_

#include <vector>
#include "./type_functor.h"

namespace tvm {
namespace relay {

/*! \brief A type visitor for vistiors which make use of internal
 * mutable state.
 *
 * We recursively visit each type contained inside the visitor.
 */
template <typename... Args>
struct TypeVisitor : ::tvm::relay::TypeFunctor<void(const Type& n, Args...)> {
  void VisitType_(const TypeParamNode* op, Args... args) override {}

  void VisitType_(const FuncTypeNode* op, Args... args) override {
    for (auto type_param : op->type_params) {
      this->VisitType(type_param, std::forward<Args>(args)...);
    }

    for (auto type_cs : op->type_constraints) {
      this->VisitType(type_cs, std::forward<Args>(args)...);
    }

    for (auto arg_type : op->arg_types) {
      this->VisitType(arg_type, std::forward<Args>(args)...);
    }
    this->VisitType(op->ret_type, std::forward<Args>(args)...);
  }

  void VisitType_(const TensorTypeNode* op, Args... args) override {}

  void VisitType_(const TupleTypeNode* op, Args... args) override {
    for (const Type& t : op->fields) {
      this->VisitType(t, std::forward<Args>(args)...);
    }
  }

  void VisitType_(const TypeRelationNode* op, Args... args) override {
    for (const Type& t : op->args) {
      this->VisitType(t, std::forward<Args>(args)...);
    }
  }

  void VisitType_(const IncompleteTypeNode* op, Args... args) override {}
};

// A functional visitor for rebuilding an AST in place.
struct TypeMutator : TypeFunctor<Type(const Type& n)> {
  Type VisitType_(const TensorTypeNode* op) override {
    // TODO(@jroesch): maybe we should recursively visit
    return TensorTypeNode::make(op->shape, op->dtype);
  }

  Type VisitType_(const TypeParamNode* op) override {
    return GetRef<TypeParam>(op);
  }

  Type VisitType_(const FuncTypeNode* op) override {
    Array<TypeParam> type_params;
    for (auto type_param : op->type_params) {
      auto new_type_param = VisitType(type_param);
      if (const TypeParamNode* tin = new_type_param.as<TypeParamNode>()) {
        type_params.push_back(GetRef<TypeParam>(tin));
      } else {
        CHECK(false) << new_type_param << std::endl;
      }
    }

    Array<TypeConstraint> type_constraints;
    for (auto type_cs : op->type_constraints) {
      auto new_type_cs = VisitType(type_cs);
      if (const TypeConstraintNode* tin =
          new_type_cs.as_derived<TypeConstraintNode>()) {
        type_constraints.push_back(GetRef<TypeConstraint>(tin));
      } else {
        CHECK(false) << new_type_cs << std::endl;
      }
    }

    std::vector<Type> args;
    for (auto arg_type : op->arg_types) {
      args.push_back(VisitType(arg_type));
    }

    return FuncTypeNode::make(tvm::Array<Type>(args), VisitType(op->ret_type),
                              type_params, type_constraints);
  }

  Type VisitType_(const TupleTypeNode* op) override {
    std::vector<Type> new_fields;
    for (const Type& t : op->fields) {
      new_fields.push_back(this->VisitType(t));
    }
    return TupleTypeNode::make(new_fields);
  }

  Type VisitType_(const TypeRelationNode* type_rel) override {
    std::vector<Type> new_args;
    for (const Type& t : type_rel->args) {
      new_args.push_back(this->VisitType(t));
    }
    return TypeRelationNode::make(type_rel->func,
                                  new_args,
                                  type_rel->num_inputs,
                                  type_rel->attrs);
  }

  Type VisitType_(const IncompleteTypeNode* op) override {
    return GetRef<Type>(op);
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_TYPE_VISITOR_H_
