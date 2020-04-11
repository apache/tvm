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
 * \file tvm/relay/expr_functor.h
 * \brief A more powerful visitor which enables defining arbitrary function
 * signatures with type based dispatch on first argument.
 */
#ifndef TVM_RELAY_EXPR_FUNCTOR_H_
#define TVM_RELAY_EXPR_FUNCTOR_H_

#include <tvm/node/functor.h>
#include <tvm/ir/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/adt.h>
#include <tvm/relay/op.h>

#include <string>
#include <utility>
#include <unordered_map>

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
      [](const ObjectRef& n, TSelf* self, Args... args) {                \
        return self->VisitExpr_(static_cast<const OP*>(n.get()), \
                                std::forward<Args>(args)...);          \
      });

template <typename R, typename... Args>
class ExprFunctor<R(const Expr& n, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const Expr& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

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
    CHECK(n.defined());
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
  virtual R VisitExpr_(const RefCreateNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const RefReadNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const RefWriteNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ConstructorNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const MatchNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExprDefault_(const Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;
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
    RELAY_EXPR_FUNCTOR_DISPATCH(RefCreateNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(RefReadNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(RefWriteNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(ConstructorNode);
    RELAY_EXPR_FUNCTOR_DISPATCH(MatchNode);
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
  void VisitExpr_(const RefCreateNode* op) override;
  void VisitExpr_(const RefReadNode* op) override;
  void VisitExpr_(const RefWriteNode* op) override;
  void VisitExpr_(const ConstructorNode* op) override;
  void VisitExpr_(const MatchNode* op) override;
  virtual void VisitType(const Type& t);
  virtual void VisitClause(const Clause& c);
  virtual void VisitPattern(const Pattern& c);

 protected:
  // Internal visiting counter
  std::unordered_map<const Object*, size_t> visit_counter_;
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
  Expr VisitExpr_(const RefCreateNode* op) override;
  Expr VisitExpr_(const RefReadNode* op) override;
  Expr VisitExpr_(const RefWriteNode* op) override;
  Expr VisitExpr_(const ConstructorNode* op) override;
  Expr VisitExpr_(const MatchNode* op) override;

  /*!
   * \brief Used to visit the types inside of expressions.
   *
   * Can be overloaded to transform the types in arbitrary
   * ways, one way would be to define a sub-class of type
   * visitor for types which transform them appropriately.
   */
  virtual Type VisitType(const Type& t);
  virtual Clause VisitClause(const Clause& c);
  virtual Pattern VisitPattern(const Pattern& c);

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, Expr, ObjectHash, ObjectEqual> memo_;
};

/*!
 * \brief A wrapper around ExprVisitor which traverses the Dataflow Normal AST.
 *
 * MixedModeVisitor treats Expr as dataflow graph, and visits in post-DFS order
 *
 * MixedModeVisitor provides the same recursive API as ExprVisitor, and uses
 * recursion to traverse most forms of the IR, but under the hood it expands nested dataflow regions
 * of the graph and processes them iteratatively to prevent stack overflows
 */
class MixedModeVisitor : public ::tvm::relay::ExprVisitor {
 public:
  /*! \brief The constructor of MixedModeVisitor
   *  \param visit_limit The number of times to allow visitation to a node. Usually 1, ocassionally
   * higher (i.e., 2 for dead code elimiation), limited to 10 as a sanity check.
   */
  explicit MixedModeVisitor(int visit_limit = 1);

  /*!
   * \brief VisitExpr is finalized to preserve call expansion of dataflow regions
   */
  void VisitExpr(const Expr& expr) final;
  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const TupleNode* op) override;
  void VisitExpr_(const TupleGetItemNode* op) override;

 protected:
  /*!
   * \brief A function to apply when reaching a leaf of the graph non-recursively
   */
  virtual void VisitLeaf(const Expr& expr);
  /*!
   * \brief A function to determine if an expression has already been visited or needs to be
   * re-visited
   */
  virtual bool CheckVisited(const Expr& expr);
  /*!
   * \brief The max number of times to visit a node
   */
  size_t visit_limit_;
};

/*! \brief Non-recursive DFS Graph Traversal for Custom Rewriting Passes
 *
 * MixedModeMutator treats Expr as dataflow graph, and only Rewrites each Expr once.
 * The mutated results are memoized in a map and reused so that
 * local transformation on the dataflow preserves the graph structure.
 *
 * MixedModeMutator provides the same recursive API as ExprMutator, and uses
 * recursion to traverse most forms of the IR, but under the hood it expands nested dataflow regions
 * of the graph and processes them iteratatively to prevent stack overflows
 *
 * Uses Rewrite_ API of ExprRewriter for a cleaner split between recrusive and non-recursive behavior.
 */
class MixedModeMutator : public ::tvm::relay::ExprMutator {
 public:
  Expr VisitExpr(const Expr& expr) final;
  virtual Expr DispatchVisitExpr(const Expr& expr);
  Expr VisitExpr_(const TupleNode* op) final { return Rewrite(op); };
  Expr VisitExpr_(const CallNode* call_node) final { return Rewrite(call_node); };
  Expr VisitExpr_(const TupleGetItemNode* op) final { return Rewrite(op); };
  /*!
   *  \brief Users should override Rewrite_ methods to implement their pass. Rewrite_ functions will be
   * able to rewrite the op only with data about the original node `pre` and the same node with
   * modified inputs `post` and should not recurse.
   *
   * \param pre The expression node before rewriting.
   * \param post The expression with rewritten inputs.
   */
  virtual Expr Rewrite_(const TupleNode* pre, const Expr& post) { return post;}
  virtual Expr Rewrite_(const CallNode* pre, const Expr& post) { return post; }
  virtual Expr Rewrite_(const TupleGetItemNode* pre, const Expr& post) { return post; }

 protected:
  /*! \brief Implement Rewrite API by calling ExprMutator's VisitExpr_(op) to get a `post` node with
   * changed inputs.
   */
  template <typename T>
  Expr Rewrite(const T* op) {
    Expr post = ExprMutator::VisitExpr_(op);
    return Rewrite_(op, post);
  }

  virtual void VisitLeaf(const Expr& expr);
  virtual bool CheckVisited(const Expr& expr);
};

#define RELAY_EXPR_REWRITER_DISPATCH(OP)                                                   \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, const Expr& post) { \
    return self->Rewrite_(static_cast<const OP*>(n.get()), post);                          \
  });

#define EXPR_REWRITER_REWRITE_DEFAULT \
  { return post; }

/*! \brief A non-iterating Expression Rewriter
 *
 *  ExprRewriter provides a Rewrite interface for modifying graphs in Post-DFS order.
 *
 * The expectation is that ExprRewriter objects will be passed to PostOrderRewrite, which will
 * non-recursively unroll the graph and call Rewriting on inputs. It will then pass the original
 * node, called `pre`, and a node recreated with any alterned inputs, called `post`, to the
 * ExprRewriter. The ExprRewriter can then use the information in those two nodes to do more complex
 * graph rewriting.
 */
class ExprRewriter {
 private:
  using TSelf = ExprRewriter;
  using FType = tvm::NodeFunctor<Expr(const ObjectRef& n, TSelf* self, const Expr& post)>;

 public:
  /*! \brief virtual destructor */
  virtual ~ExprRewriter() {}
  /*!
   * \brief Same as call.
   * \param pre The expression node before rewriting.
   * \param post The expression node with rewritten inputs.
   * \return The result of the call
   */
  Expr operator()(const Expr& pre, const Expr& post) {
    return Rewrite(pre, post);
  }
  /*!
   * \brief The functor call.
   * \param pre The expression node before rewriting.
   * \param post The expression node with rewritten inputs.
   * \return The result of the call
   */
  virtual Expr Rewrite(const Expr& pre, const Expr& post) {
    CHECK(pre.defined());
    static FType vtable = InitVTable();
    return vtable(pre, this, post);
  }
  // Functions that can be overriden by subclass, should not recurse
  virtual Expr Rewrite_(const VarNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const GlobalVarNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const ConstantNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const TupleNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const FunctionNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const CallNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const LetNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const IfNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const OpNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const TupleGetItemNode* pre,
                        const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const RefCreateNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const RefReadNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const RefWriteNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const ConstructorNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;
  virtual Expr Rewrite_(const MatchNode* pre, const Expr& post) EXPR_REWRITER_REWRITE_DEFAULT;

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAY_EXPR_REWRITER_DISPATCH(ConstantNode);
    RELAY_EXPR_REWRITER_DISPATCH(TupleNode);
    RELAY_EXPR_REWRITER_DISPATCH(VarNode);
    RELAY_EXPR_REWRITER_DISPATCH(GlobalVarNode);
    RELAY_EXPR_REWRITER_DISPATCH(FunctionNode);
    RELAY_EXPR_REWRITER_DISPATCH(CallNode);
    RELAY_EXPR_REWRITER_DISPATCH(LetNode);
    RELAY_EXPR_REWRITER_DISPATCH(IfNode);
    RELAY_EXPR_REWRITER_DISPATCH(OpNode);
    RELAY_EXPR_REWRITER_DISPATCH(TupleGetItemNode);
    RELAY_EXPR_REWRITER_DISPATCH(RefCreateNode);
    RELAY_EXPR_REWRITER_DISPATCH(RefReadNode);
    RELAY_EXPR_REWRITER_DISPATCH(RefWriteNode);
    RELAY_EXPR_REWRITER_DISPATCH(ConstructorNode);
    RELAY_EXPR_REWRITER_DISPATCH(MatchNode);
    return vtable;
  }
};

/*! \brief Non-recursive DFS Graph Traversal for Custom Rewriting Passes
 *
 * PostOrderRewrite does a non-recursive traversal of the graph in Post-DFS order and calls the
 * ExprRewriter's Rewrite functions on nodes once their inputs are rewritten. At each rewrite call,
 * PostOrderRewrite provides the original node and the node with altered inputs for use by the
 * ExprRewriter.
 */
Expr PostOrderRewrite(const Expr& expr, ExprRewriter* rewriter);

/*!
 * \brief recursively visit the ir in post DFS order node, apply fvisit
 * Each node is guaranteed to be visited only once.
 * \param node The ir to be visited.
 * \param fvisit The visitor function to be applied.
 */
void PostOrderVisit(const Expr& node, std::function<void(const Expr&)> fvisit);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_EXPR_FUNCTOR_H_
