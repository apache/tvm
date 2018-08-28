/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/expr_functor.h
 * \brief A more powerful visitor which enables defining arbitrary function
 * signatures with type based dispatch on first argument.
 */
#ifndef TVM_RELAY_EXPR_FUNCTOR_H_
#define TVM_RELAY_EXPR_FUNCTOR_H_

#include <tvm/ir_functor.h>
#include <string>
#include "./expr.h"
#include "./op.h"

namespace tvm {
namespace relay {

/*!
 * \brief A dynamical functor that dispatches on in the first Expr argument.
 *  You can use this as a more powerful Visitor, since it allows you to
 *  define function signatures of Visit Function.
 * 
 * \sa tvm/ir_functor.h
 *
 * \tparam FType function signiture
 *  This type if only defined for FType with function signiture R(const Expr&,
 * Args...)
 */
template <typename FType>
class ExprFunctor;

// functions to be overriden.
#define EXPR_FUNCTOR_DEFAULT \
  { return VisitExprDefault_(op, std::forward<Args>(args)...); }

#define RELAY_EXPR_FUNCTOR_DISPATCH(OP)                                \
  vtable.template set_dispatch<OP>(                                    \
      [](const NodeRef& n, TSelf* self, Args... args) {                \
        return self->VisitExpr_(static_cast<const OP*>(n.node_.get()), \
                                std::forward<Args>(args)...);          \
      });

template <typename R, typename... Args>
class ExprFunctor<R(const Expr& n, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const Expr& n, Args...)>;
  using FType = tvm::IRFunctor<R(const NodeRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~ExprFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Expr& n, Args... args) {
    return VisitExpr(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitExpr(const Expr& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitExpr_(const ConstantNode* op,
                       Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleNode* op,
                       Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const LocalVarNode* op,
                       Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const GlobalVarNode* op,
                       Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ParamNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FunctionNode* op,
                       Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const CallNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const LetNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const IfNode* op,
                       Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const OpNode* op,
                       Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExprDefault_(const Node* op, Args...) {
    throw dmlc::Error(std::string("Do not have a default for ") + op->type_key());
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAY_EXPR_FUNCTOR_DISPATCH(ConstantNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(TupleNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(LocalVarNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(GlobalVarNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(ParamNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(FunctionNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(CallNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(LetNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(IfNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(OpNode);
    return vtable;
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_EXPR_FUNCTOR_H_
