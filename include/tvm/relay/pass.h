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
 * \file tvm/relay/pass.h
 * \brief The set of Relay passes written in C++.
 *
 * This file also implements a pass manager. The pass manager manages a sequence
 * of Relay-to-Relay transformation passes over a particlar unit of AST. The
 * design is largely inspired from LLVM's pass manager and modern deep learning
 * frameworks that perform tensor->tensor transformations.
 *
 * The responsibilities of a traditional compiler pass manager usually involves:
 *  - Organizing the execution order of optimization passes though not
 * necessarily in the optimal sequence.
 *  - Collecting required analysis information and keep them up-to-date.
 *  - Reducing the effort required to implement new passes for compiler
 * developers, etc.
 *
 * Similar to LLVM's pass manager, we designed the Relay pass manager to work
 * different granularity, i.e. module level, function level, and even sequential
 * passe that contains a host of passes.
 *
 * However, we also extend the functionality of the traditional pass manager
 * with the consideration of requirements/convention from deep learning
 * frameworks, such as Pytorch and Gluon, etc. Each pass in the Relay pass
 * manager performs the Relay.Module -> Relay.Module transformation. All
 * different types of passes, including the sequential-level pass object, are
 * essentially pass objects. This design, therefore, effectively provides users
 * a consistent and convenient interface, i.e. Pass, to play with. It offers a
 * means to ease the development and testing of Relay passes. For example, with
 * the pass manager, external users will be able to have custom passes correctly
 * scheduled without having to modify a single handcrafted pass order.
 *
 * In the future we need to describe constraints between passes. For example,
 * we may want to preserve dependencies between different passes and validate
 * them on the completion of a certain pass.
 *
 * We also need to store side information and import the error reporting system.
 */
#ifndef TVM_RELAY_PASS_H_
#define TVM_RELAY_PASS_H_

#include <tvm/ir.h>
#include <tvm/packed_func_ext.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/module.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/type.h>
#include <tvm/relay/adt.h>
#include <string>
#include <vector>

namespace tvm {
namespace relay {

namespace pass {

/*
 * \brief The context of pass.
 */
class PassContext;

/*!
 * \brief PassContextNode contains the information that a pass can rely on, such as
 * analysis results.
 */
class PassContextNode : public RelayNode {
 public:
  /*!
   * \brief The error reporter used to notify users why an optimization fails.
   */
  ErrorReporter err_reporter;

  PassContextNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
  }

  TVM_DLL static PassContext make();

  static constexpr const char* _type_key = "relay.PassContext";
  TVM_DECLARE_NODE_TYPE_INFO(PassContextNode, RelayNode);
};

TVM_DEFINE_NODE_REF(PassContext, PassContextNode)

/*
 * \brief The meta data of a pass.
 *
 * PassInfo can be extended conveniently in the future if more meta information
 * is needed.
 */
class PassInfo;

/*!
 * \brief PassInfoNode contains meta data that will be used to help optimization
 * and analysis.
 */
class PassInfoNode : public RelayNode {
 public:
  /*! \brief The minimal optimization level that this pass will be enabled. */
  int opt_level;

  /*! \brief The name of an optimization/analysis pass. */
  std::string name;

  /*! \brief The passes that are required to perform the current pass. */
  tvm::Array<tvm::Expr> required;

  PassInfoNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("opt_level", &opt_level);
    v->Visit("name", &name);
    v->Visit("required", &required);
  }

  TVM_DLL static PassInfo make(int opt_level, std::string name,
                               tvm::Array<tvm::Expr> required);

  static constexpr const char* _type_key = "relay.PassInfo";
  TVM_DECLARE_NODE_TYPE_INFO(PassInfoNode, RelayNode);
};

TVM_DEFINE_NODE_REF(PassInfo, PassInfoNode)

class Pass;

/*!
 * \brief PassNode is the base type of differnt types of optimization passes.
 * It is designed as a pure class and implemented by different pass subclasses
 * at different granularity of Relay nodes.
 */
class PassNode : public RelayNode {
 public:
  /*
   * \brief Get the pass information/meta data. */
  virtual PassInfo Info() const = 0;

  /*!
   * \brief Set the context information for a pass.
   *
   * \param pass_ctx The context information for a certain pass.
   */
  virtual void SetContext(const PassContext& pass_ctx) = 0;

  /*!
   * \brief Execute the optimization pass using a functor.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return The updated module.
   */
  virtual Module operator()(const Module& mod) const = 0;

  void VisitAttrs(tvm::AttrVisitor* v) override {}

  static constexpr const char* _type_key = "relay.Pass";
  TVM_DECLARE_BASE_NODE_INFO(PassNode, RelayNode);
};

class Pass : public NodeRef {
 public:
  Pass() = default;
  explicit Pass(NodePtr<tvm::Node> p) : NodeRef(p) {}

  PassNode* operator->() const {
    return static_cast<PassNode*>(this->node_.get());
  }

  using ContainerType = PassNode;
};

/*
 * \brief Create a module pass.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the module pass.
 * \param name The name of the module pass.
 * \param required The list of the passes that the module pass is dependent on.
 *
 * \return The created module pass.
 */
Pass CreateModulePass(
    const runtime::TypedPackedFunc<Module(Module, PassContext)>& pass_func,
    int opt_level,
    const std::string& name,
    const tvm::Array<tvm::Expr>& required);

/*
 * \brief Create a function pass.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the function pass.
 * \param name The name of the function pass.
 * \param required The list of the passes that the function pass is dependent on.
 *
 * \return The created function pass.
 */
Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, PassContext)>& pass_func,
    int opt_level,
    const std::string& name,
    const tvm::Array<tvm::Expr>& required);
/*
 * \brief Create a sequential pass.
 *
 * \param passes The optimization passes will be performed.
 * \param opt_level The optimization level of the sequential pass.
 * \param name The name of the sequential pass.
 * \param required The list of the passes that the sequential pass is dependent on.
 * \param disabled The disabled passes.
 *
 * \return The created sequential pass.
 */
Pass CreateSequentialPass(const tvm::Array<Pass>& passes,
                          int opt_level,
                          const std::string& name,
                          const tvm::Array<tvm::Expr>& required,
                          const tvm::Array<tvm::Expr>& disabled);

}  // namespace pass

/*!
 * \brief Infer the type of an expression.
 *
 * The result of type checking is a new expression with unambigous
 * type information filled in, as well as it's checked type field
 * populated with the result type.
 *
 * \param expr The expression to type check.
 * \param mod The module used for referencing global functions, can be
 * None.
 *
 * \return A type checked expression with its checked_type field populated.
 */
TVM_DLL Expr InferType(const Expr& expr, const Module& mod);

/*!
 * \brief Infer the type of a function as if it is mapped to var in the mod.
 *
 * \param f the function.
 * \param mod The module used for referencing global functions.
 * \param var The global variable corresponding to the function.
 *
 * \return A type checked Function with its checked_type field populated.
 * \note this function mutates mod and is not thread-safe.
 */
TVM_DLL Function InferType(const Function& f, const Module& mod,
                           const GlobalVar& var);

/*!
 * \brief Check that types are well kinded by applying "kinding rules".
 *
 * This pass ensures we do not do things that violate the design of the
 * type system when writing down types.
 *
 * For example tensors are not allowed to contain functions in Relay.
 *
 * We check this by ensuring the `dtype` field of a Tensor always contains
 * a data type such as `int`, `float`, `uint`.
 *
 * \param t The type to check.
 * \param mod The global module.
 *
 * \return The kind of the passed type.
 */
TVM_DLL Kind KindCheck(const Type& t, const Module& mod);

/*! \brief Compare two expressions for structural equivalence.
 *
 * This comparison operator respects scoping and compares
 * expressions without regard to variable choice.
 *
 * For example: `let x = 1 in x` is equal to `let y = 1 in y`.
 *
 *   See https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence
 *   for more details.
 *
 *   \param e1 The left hand expression.
 *   \param e2 The right hand expression.
 *
 *   \return true if equal, otherwise false
 */
TVM_DLL bool AlphaEqual(const Expr& e1, const Expr& e2);

/*! \brief Compare two types for structural equivalence.
 *
 * This comparison operator respects scoping and compares
 * expressions without regard to variable choice.
 *
 * For example: `forall s, Tensor[f32, s]` is equal to
 * `forall w, Tensor[f32, w]`.
 *
 * See https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence
 * for more details.
 *
 * \param t1 The left hand type.
 * \param t2 The right hand type.
 *
 * \return true if equal, otherwise false
 */
TVM_DLL bool AlphaEqual(const Type& t1, const Type& t2);

/*! \brief Add abstraction over a function
 *
 * For example: `square` is transformed to
 * `fun x -> square x`.
 *
 * See https://en.wikipedia.org/wiki/Lambda_calculus#%CE%B7-conversion
 * for more details.
 *
 * \param e The original function.
 * \param mod The module used for referencing global functions, can be
 * None.
 *
 * \return the new function with abstraction
 */
TVM_DLL Expr EtaExpand(const Expr& e, const Module& mod);

/*! \brief Check that each Var is only bound once.
 *
 * For example, the expression `let x = 1 in let x = 2 in 3` bound x twice.
 *
 * `let f = (\x -> x) in let g = (\x -> x + 1) in f(g(2))` also bound x twice,
 * although x is not shadowed.
 *
  * \param expr the expression to check.
 *
  * \return true iff all Var in expr is bound at most once.
 */
TVM_DLL bool WellFormed(const Expr& expr);

/*! \brief Get all bound variables from expression expr.
 *
 * Bound variables are all variables that are declared in the expr.
 * They only have meaning inside that expr, and can only be used in it.
 *
 * \param expr the expression.
 *
 * \return List of bound vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> BoundVars(const Expr& expr);

/*! \brief Get all bound variables from pattern pat.
 *
 * Bound variables are all variables that got bound by the pat.
 * They only have meaning inside that expr, and can only be used in it.
 *
 * \param pat the Pattern.
 *
 * \return List of bound vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> BoundVars(const Pattern& pat);

/*! \brief Get free type parameters from expression expr.
 *
 * Free variables are variables that are not bound by a
 * let or a function parameter in the context.
 *
 * \param expr the expression.
 *
 * \return List of free vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> FreeVars(const Expr& expr);

/*! \brief Get all variables from expression expr.
 *
 * \param expr the expression.
 *
 * \return List of all vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> AllVars(const Expr& expr);

/*! \brief Get free TypeVars from expression expr.
 *
 * Free type parameters are type parameters that are not bound by a function
 * type in the context.
 *
 * \param expr the expression.
 * \param mod the module.
 *
 * \return List of free vars, in the PostDFS order visited by expr.
 */
TVM_DLL tvm::Array<TypeVar> FreeTypeVars(const Expr& expr, const Module& mod);

/*! \brief Get free TypeVars from type t.
 *
 * Free type parameters are type parameters that are not bound by a function
 * type in the context.
 *
 * \param t the type.
 * \param mod the module.
 *
 * \return List of free type vars, in the PostDFS order visited by type.
 */
TVM_DLL tvm::Array<TypeVar> FreeTypeVars(const Type& t, const Module& mod);

/*! \brief Get all bound type variables from expression expr.
 *
 * Bound variables are all type variables that are declared in the expr.
 * They only have meaning inside that expr, and can only be used in it.
 *
 * \param expr the expression.
 * \param mod the module.
 *
 * \return List of bound type vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<TypeVar> BoundTypeVars(const Expr& expr, const Module& mod);

/*! \brief Get all bound type variables from type t.
 *
 * Bound variables are all type variables that are declared in the type.
 * They only have meaning inside that type, and can only be used in it.
 *
 * \param t the type
 * \param mod the module.
 *
 * \return List of bound type vars, in the PostDFS order visited by type.
 */
TVM_DLL tvm::Array<TypeVar> BoundTypeVars(const Type& t, const Module& mod);

/*! \brief Get all type variables in expression expr.
 *
 * \param expr the expression.
 * \param mod the module.
 *
 * \return List of type vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<TypeVar> AllTypeVars(const Expr& expr, const Module& mod);

/*! \brief Get all type variables in type t.
 *
 * \param t the type.
 * \param mod the module.
 *
 * \return List of type vars, in the PostDFS order visited by type.
 */
TVM_DLL tvm::Array<TypeVar> AllTypeVars(const Type& t, const Module& mod);

/*! \brief Remove expressions which does not effect the program result.
 *
 * It will remove let bindings which are not referenced,
 * and inline let bindings that are only used once.
 *
 * For example, this pass should turn `let a = 1 in 2` into `2`,
 * as the value of the expression does not depend on a.
 *
 * As another example, `let a = 1 in a` will be optimized into 1.
 *
 * \param e the expression to optimize.
 *
 * \return the optimized expression.
 */
TVM_DLL Expr DeadCodeElimination(const Expr& e);

/*!
 * \brief Fold constant expressions.
 * \param expr the expression to be optimized.
 * \return The optimized expression.
 */
TVM_DLL Expr FoldConstant(const Expr& expr);

/*!
 * \brief Fuse operations into expr into seperate functions.
 * \param expr The expression.
 * \param fuse_opt_level Optimization level.
 * \param mod the module.
 * \return The optimized expression.
 */
TVM_DLL Expr FuseOps(const Expr& expr, int fuse_opt_level, const Module& mod);

/*!
 * \brief Apply rewrite rules to rewrite the expr in post DFS order.
 * \param expr The expression.
 * \param rewrite_map_attr_name The Op's attr name which corresponds to the rewrite
 *                              rule function.
 * \param fcontext Additional callback to provide context argument for each call node.
 * \param fmulti_ref_trigger Transformation function to be called when
 *                           an Expr consumed by multiple callers.
 * \return The rewritten expression.
 */
TVM_DLL Expr ForwardRewrite(const Expr& expr,
                    const std::string& rewrite_map_attr_name,
                    std::function<NodeRef(const Call&)> fcontext = nullptr,
                    std::function<Expr(const Expr&)> fmulti_ref_trigger = nullptr);

/*!
 * \brief Apply rewrite rules to rewrite the expr in post DFS order.
 * \param expr The expression.
 * \param rewrite_func The rewrite func that will apply to all operators.
 * \param fcontext Additional callback to provide context argument for each call node.
 * \param fmulti_ref_trigger Transformation function to be called when
 *                           an Expr consumed by multiple callers.
 * \return The rewritten expression.
 */
TVM_DLL Expr ForwardRewrite(const Expr& expr,
                    const FForwardRewrite& rewrite_func,
                    std::function<NodeRef(const Call&)> fcontext = nullptr,
                    std::function<Expr(const Expr&)> fmulti_ref_trigger = nullptr);

/*!
 * \brief Rewrite the annotated program.
 * \param expr The expression.
 * \param fallback_device The fallback device which is the default device for
 *                        operators without annotation.
 * \return The updated program.
 */
TVM_DLL Expr RewriteAnnotatedOps(const Expr& expr, int fallback_device);

/*!
 * \brief Collect the device mapping information of each expression.
 * \param expr The expression.
 * \return The device mapping.
 */
TVM_DLL Map<Expr, Integer> CollectDeviceInfo(const Expr& expr);

/*! \brief A hashing structure in the style of std::hash. */
struct StructuralHash {
  /*! \brief Hash a Relay type.
   *
   * Implements structural hashing of a Relay type.
   *
   *  \param type the type to hash.
   *
   *  \return the hash value.
   */
  size_t operator()(const Type& type) const;

  /*! \brief Hash a Relay expression.
   *
   * Implements structural hashing of a Relay expression.
   *
   * \param expr the expression to hash.
   *
   * \return the hash value.
   */
  size_t operator()(const Expr& expr) const;
};

/*! \brief turn a dataflow graph into Administrative Normal Form, or A-Normal Form (ANF).
 *
 * It will turn an expression that is in a graph form (with sharing implicit),
 * to an expression with explicit sharing (A-Normal Form).
 *
 * The scope of the root expression is the global scope.

 * The scope of any non root expression is the least common ancestor of all it's scope.
 *
 * Values are ordered by post-DFS order in each scope.
 *
 * \param e the expression to observably share
 *
 * \param mod The module used for referencing global functions, can be
 * None.
 *
 * \return expression in A-Normal Form
 */
TVM_DLL Expr ToANormalForm(const Expr& e, const Module& mod);

/*! \brief Remove let binding and directly share via pointer instead.
 *
 * It will remove all let binding,
 * and turn all of the variable bound by let into direct pointer reference.
 *
 * \param e the expression.
 *
 * \return the expression in graph normal form.
 */
TVM_DLL Expr ToGraphNormalForm(const Expr& e);

/*! \brief Aggressive constant propagation/constant folding/inlining.
 * It will do as much computation in compile time as possible.
 * It has two benefit: remove runtime overhead, and allow more optimization (typically fusion).
 * As a side effect, code size will explode.
 */
Expr PartialEval(const Expr& e);
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PASS_H_
