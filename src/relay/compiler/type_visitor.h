/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_visitor.h
 * \brief A wrapper around TypeFunctor for common use cases.
 */
#ifndef TVM_RELAY_TYPE_VISITOR_H_
#define TVM_RELAY_TYPE_VISITOR_H_

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
    // fix me handle poly
    // this->VisitType(op->var, args...);
    // this->VisitType(op->boundType, args...);
    for (auto arg_type : op->arg_types) {
      this->VisitType(arg_type, args...);
    }
    this->VisitType(op->ret_type, args...);
  }

  void VisitType_(const TensorTypeNode* op, Args... args) override {}

  //   void VisitType_(const TupleTypeNode* op, Args... args) override {
  //     for (const Type& t : op->fields) {
  //       this->VisitType(t, args...);
  //     }
  //   }

  void VisitType_(const TypeCallNode* op, Args... args) override {
    this->VisitType(op->func, args...);

    for (const Type& t : op->args) {
      this->VisitType(t, args...);
    }
  }

  void VisitType_(const TypeFunctionNode* op, Args... args) override {}
  void VisitType_(const IncompleteTypeNode* op, Args... args) override {}
};

// A functional visitor for rebuilding an AST in place.
struct TypeFVisitor : TypeFunctor<Type(const Type& n)> {
  Type VisitType_(const TensorTypeNode* op) override {
    // TODO (@jroesch): maybe we should recursively visit
    return TensorTypeNode::make(op->shape, op->dtype);
  }

  Type VisitType_(const TypeParamNode* op) override {
    return GetRef<TypeParam>(op);
  }

  Type VisitType_(const FuncTypeNode* op) override {
    // auto new_id = this->VisitType(op->var);
    // if (const TypeParamNode* tin = new_id.as<TypeParamNode>()) {
    // return TypeQuantifierNode::make(GetRef<TypeParam>(tin),
    //                                this->VisitType(op->boundType));

      std::vector<Type> args;
      for (auto arg_type : op->arg_types) {
        args.push_back(VisitType(arg_type));
      }

      return FuncTypeNode::make(tvm::Array<Type>(args),
                                 VisitType(op->ret_type), {}, {}); // fix me
    }

    //   Type VisitType_(const TupleTypeNode* op) override {
    //     std::vector<Type> new_fields;
    //     for (const Type& t : op->fields) {
    //       new_fields.push_back(this->VisitType(t));
    //     }
    //     return TupleTypeNode::make(new_fields);
    //   }

    Type VisitType_(const TypeCallNode* op) override {
      auto func = this->VisitType(op->func);
      std::vector<Type> new_args;
      for (const Type& t : op->args) {
        new_args.push_back(this->VisitType(t));
      }
      return TypeCallNode::make(func, new_args);
    }

    Type VisitType_(const IncompleteTypeNode* op) override {
      return GetRef<IncompleteType>(op);
    }
  };

}  // namespace relay
}  // namespace relay
#endif  // TVM_RELAY_TYPE_VISITOR_H_
