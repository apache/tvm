/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_functor.h
 * \brief A way to defined arbitrary function signature with dispatch on types.
 */
#ifndef TVM_RELAY_IR_TYPE_FUNCTOR_H_
#define TVM_RELAY_IR_TYPE_FUNCTOR_H_

#include <tvm/node/ir_functor.h>
#include <tvm/relay/expr.h>
#include <string>
#include <vector>

namespace tvm {
namespace relay {

template <typename FType>
class TypeFunctor;

// functions to be overriden.
#define TYPE_FUNCTOR_DEFAULT \
  { return VisitTypeDefault_(op, std::forward<Args>(args)...); }


#define RELAY_TYPE_FUNCTOR_DISPATCH(OP)                                   \
  vtable.template set_dispatch<OP>(                                       \
      [](const NodeRef& n, TSelf* self, Args... args) {                   \
        return self->VisitType_(static_cast<const OP*>(n.node_.get()),    \
                                std::forward<Args>(args)...);             \
      });

template <typename R, typename... Args>
class TypeFunctor<R(const Type& n, Args...)> {
 private:
  using TSelf = TypeFunctor<R(const Type& n, Args...)>;
  using FType = tvm::IRFunctor<R(const NodeRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~TypeFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Type& n, Args... args) {
    return VisitType(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitType(const Type& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitType_(const TensorTypeNode* op,
                       Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeVarNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeConstraintNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const FuncTypeNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeRelationNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TupleTypeNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const IncompleteTypeNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;

  virtual R VisitTypeDefault_(const Node* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->type_key();
    throw;  // unreachable, written to stop compiler warning
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAY_TYPE_FUNCTOR_DISPATCH(TensorTypeNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(TypeVarNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(TypeConstraintNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(FuncTypeNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(TypeRelationNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(TupleTypeNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(IncompleteTypeNode);
    return vtable;
  }
};

/*!
 * \brief A type visitor for vistiors which make use of internal
 * mutable state.
 *
 * We recursively visit each type contained inside the visitor.
 */
class TypeVisitor :
    public ::tvm::relay::TypeFunctor<void(const Type& n)> {
 public:
  void VisitType_(const TypeVarNode* op) override {}

  void VisitType_(const FuncTypeNode* op) override {
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

  void VisitType_(const TensorTypeNode* op) override {}

  void VisitType_(const TupleTypeNode* op) override {
    for (const Type& t : op->fields) {
      this->VisitType(t);
    }
  }

  void VisitType_(const TypeRelationNode* op) override {
    for (const Type& t : op->args) {
      this->VisitType(t);
    }
  }

  void VisitType_(const IncompleteTypeNode* op) override {}
};

// A functional visitor for rebuilding an AST in place.
struct TypeMutator : TypeFunctor<Type(const Type& n)> {
  Type VisitType_(const TensorTypeNode* op) override {
    // TODO(@jroesch): maybe we should recursively visit
    return TensorTypeNode::make(op->shape, op->dtype);
  }

  Type VisitType_(const TypeVarNode* op) override {
    return GetRef<TypeVar>(op);
  }

  Type VisitType_(const FuncTypeNode* op) override {
    Array<TypeVar> type_params;
    for (auto type_param : op->type_params) {
      auto new_type_param = VisitType(type_param);
      if (const TypeVarNode* tin = new_type_param.as<TypeVarNode>()) {
        type_params.push_back(GetRef<TypeVar>(tin));
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
#endif  // TVM_RELAY_IR_TYPE_FUNCTOR_H_
