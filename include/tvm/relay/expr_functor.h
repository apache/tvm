/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/expr_functor.h
 * \brief A more powerful visitor which enables defining arbitrary function
 * signatures with type based dispatch on first argument.
 */
#ifndef TVM_RELAY_EXPR_FUNCTOR_H_
#define TVM_RELAY_EXPR_FUNCTOR_H_

#include <tvm/node/ir_functor.h>
#include <string>
#include "./expr.h"
#include "./op.h"
#include "./error.h"

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
 *  This type is only defined for FType with function signature R(const Expr&,
 * Args...)
 */
template <typename FType>
class ExprFunctor;

// functions to be overriden.
#define EXPR_FUNCTOR_DEFAULT                                      \
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
  virtual R VisitExpr_(const VarNode* op,
                       Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const GlobalVarNode* op,
                       Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FunctionNode* op,
                       Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const CallNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const LetNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const IfNode* op,
                       Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const OpNode* op,
                       Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleGetItemNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExprDefault_(const Node* op, Args...) {
    throw Error(std::string("Do not have a default for ") + op->type_key());
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAY_EXPR_FUNCTOR_DISPATCH(ConstantNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(TupleNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(VarNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(GlobalVarNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(FunctionNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(CallNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(LetNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(IfNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(OpNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(TupleGetItemNode);
    return vtable;
  }
};

/*!
 * \brief A simple visitor wrapper around ExprFunctor.
 *  Recursively visit the content.
 *
 * ExprVisitor treats Expr as dataflow graph,
 * and only visit each Expr node once.
 */
class ExprVisitor
    : public ::tvm::relay::ExprFunctor<void(const Expr& n)> {
 public:
  void VisitExpr(const Expr& expr) override;
  void VisitExpr_(const VarNode* op) override;
  void VisitExpr_(const GlobalVarNode* op) override;
  void VisitExpr_(const ConstantNode* op) override;
  void VisitExpr_(const TupleNode* op) override;
  void VisitExpr_(const FunctionNode* op) override;
  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const LetNode* op) override;
  void VisitExpr_(const IfNode* op) override;
  void VisitExpr_(const OpNode* op) override;
  void VisitExpr_(const TupleGetItemNode* op) override;
  virtual void VisitType(const Type& t);

 protected:
  // Internal visiting counter
  std::unordered_map<const Node*, size_t> visit_counter_;
};

/*!
 * \brief A wrapper around ExprFunctor which functionally updates the AST.
 *
 * ExprMutator treats Expr as dataflow graph, and only Mutate each Expr once.
 * The mutated results are memoized in a map and reused so that
 * local transformation on the dataflow preserves the graph structure.
 */
class ExprMutator
    : public ::tvm::relay::ExprFunctor<Expr(const Expr&)> {
 public:
  /*!
   * \brief Mutate is alias for VisitExpr
   * \return expr.
   */
  Expr Mutate(const Expr& expr) {
    return this->VisitExpr(expr);
  }
  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const VarNode* op) override;
  Expr VisitExpr_(const ConstantNode* op) override;
  Expr VisitExpr_(const GlobalVarNode* op) override;
  Expr VisitExpr_(const OpNode* op) override;
  Expr VisitExpr_(const TupleNode* op) override;
  Expr VisitExpr_(const FunctionNode* op) override;
  Expr VisitExpr_(const CallNode* call_node) override;
  Expr VisitExpr_(const LetNode* op) override;
  Expr VisitExpr_(const IfNode* op) override;
  Expr VisitExpr_(const TupleGetItemNode* op) override;
  /*!
   * \brief Used to visit the types inside of expressions.
   *
   * Can be overloaded to transform the types in arbitrary
   * ways, one way would be to define a sub-class of type
   * visitor for types which transform them appropriately.
   */
  virtual Type VisitType(const Type& t);

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, Expr, NodeHash, NodeEqual> memo_;
};

/*!
 * \brief recursively visit the ir in post DFS order node, apply fvisit
 * Each node is guaranteed to be visited only once.
 * \param node The ir to be visited.
 * \param fvisit The visitor function to be applied.
 */
void PostOrderVisit(const NodeRef& node, std::function<void(const NodeRef&)> fvisit);

/*
 * \brief Bind function parameters or free variables.
 *
 * Parameter binding can only happen if expr is a Function.
 * binds cannot change internal arguments of internal functions.
 *
 * \param expr The function to be binded.
 * \param binds The map of arguments to
 */
Expr Bind(const Expr& expr, const tvm::Map<Var, Expr>& binds);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_EXPR_FUNCTOR_H_
