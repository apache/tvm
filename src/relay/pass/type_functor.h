/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_functor.h
 * \brief A way to defined arbitrary function signature with dispatch on types.
 */
#ifndef TVM_RELAY_PASS_TYPE_FUNCTOR_H_
#define TVM_RELAY_PASS_TYPE_FUNCTOR_H_

#include <tvm/node/ir_functor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/error.h>
#include <string>

namespace tvm {
namespace relay {

template <typename FType>
class TypeFunctor;

// functions to be overriden.
#define TYPE_FUNCTOR_DEFAULT \
  { return VisitTypeDefault_(op, std::forward<Args>(args)...); }

#define RELAY_TYPE_FUNCTOR_DISPATCH(OP)                       \
  vtable.template set_dispatch<OP>(                           \
      [](const NodeRef& n, TSelf* self, Args... args) {       \
        return self->VisitType_(static_cast<const OP*>(n.node_.get()),    \
                                std::forward<Args>(args)...); \
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
  virtual R VisitType_(const TypeParamNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
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
    RELAY_TYPE_FUNCTOR_DISPATCH(TypeParamNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(TypeConstraintNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(FuncTypeNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(TypeRelationNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(TupleTypeNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(IncompleteTypeNode);
    return vtable;
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_TYPE_FUNCTOR_H_
