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
 * \file tvm/relay/transform.h
 *
 * This file implements a pass manager. The pass manager manages a sequence
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
#ifndef TVM_RELAY_TRANSFORM_H_
#define TVM_RELAY_TRANSFORM_H_

#include <tvm/base.h>
#include <tvm/packed_func_ext.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/module.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relay {
namespace transform {

/*
 * \brief The context of pass.
 */
class PassContext;

/*!
 * \brief PassContextNode contains the information that a pass can rely on,
 * such as analysis results.
 */
class PassContextNode : public RelayNode {
 public:
  /*!
   * \brief The error reporter used to notify users why an optimization fails.
   */
  ErrorReporter err_reporter;

  /*! \brief The default optimization level. */
  int opt_level{2};

  /*! \brief CPU is the default fallback device for heterogeneous execution. */
  int fallback_device{static_cast<int>(kDLCPU)};

  /*! \brief The list of required passes. */
  tvm::Array<tvm::Expr> required_pass;
  /*! \brief The list of disabled passes. */
  tvm::Array<tvm::Expr> disabled_pass;

  PassContextNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("opt_level", &opt_level);
    v->Visit("fallback_device", &fallback_device);
    v->Visit("required_pass", &required_pass);
    v->Visit("disabled_pass", &disabled_pass);
  }

  static constexpr const char* _type_key = "relay.PassContext";
  TVM_DECLARE_NODE_TYPE_INFO(PassContextNode, RelayNode);
};

/*!
 * \brief PassContext that is used to configure the pass behavior.
 *
 * \code
 *
 *  auto new_ctx = PassContext::Create();
 *  ctx->opt_level = 2;
 *  ctx->fallback_device = kDLCPU;
 *  With<PassContext> scope(ctx);
 *  // pass context in effect.
 *
 * \endcode
 */
class PassContext : public NodeRef {
 public:
  PassContext() {}
  explicit PassContext(NodePtr<::tvm::Node> n) : NodeRef(n) {}
  /*!
   * \brief const accessor.
   * \return const access pointer.
   */
  const PassContextNode* operator->() const {
    CHECK(node_.get() != nullptr);
    return static_cast<const PassContextNode*>(node_.get());
  }
  /*!
   * \brief mutable accessor.
   * \return mutable access pointer.
   */
  PassContextNode* operator->() {
    CHECK(node_.get() != nullptr);
    return static_cast<PassContextNode*>(node_.get());
  }
  /*!
   * \brief Construct a PassContext containing the default configurations.
   * \return The new PassContext.
   */
  TVM_DLL static PassContext Create();
  /*!
   * \brief Get the default pass context in the current scope.
   * \return The pass context.
   */
  TVM_DLL static PassContext Current();

  // accessor.
  using ContainerType = PassContextNode;
  class Internal;

 private:
  // The entry of a pass context scope.
  TVM_DLL void EnterWithScope();
  // The exit of a pass context scope.
  TVM_DLL void ExitWithScope();

  // Classes to get the Python `with` like syntax.
  friend class Internal;
  friend class tvm::With<PassContext>;
};

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

  TVM_DLL static PassInfo make(int opt_level,
                               std::string name,
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
  /*!
   * \brief Get the pass information/meta data. */
  virtual PassInfo Info() const = 0;

  /*!
   * \brief Transform mod using the default PassContext in the current scope.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return The transformed module.
   */
  Module operator()(const Module& mod) const {
    return this->operator()(mod, PassContext::Current());
  }

  /*!
   * \brief Transform mod using a functor under a given pass context.
   *
   * \param mod The module that an optimization pass runs on.
   * \param pass_ctx The pass context that can provide information for the optimization.
   *
   * \return The transformed module.
   */
  virtual Module operator()(const Module& mod,
                            const PassContext& pass_ctx) const = 0;

  void VisitAttrs(tvm::AttrVisitor* v) override {}

  static constexpr const char* _type_key = "relay.Pass";
  TVM_DECLARE_BASE_NODE_INFO(PassNode, RelayNode);
};

class Pass : public NodeRef {
 public:
  /*!
   * \brief Transform mod using the default PassContext in the current scope.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return The transformed module.
   */
  Module operator()(const Module& mod) const {
    const PassNode* node = operator->();
    CHECK(node != nullptr);
    return node->operator()(mod);
  }
  /*!
   * \brief Transform mod using a functor under a given pass context.
   *
   * \param mod The module that an optimization pass runs on.
   * \param pass_ctx The pass context that can provide information for the optimization.
   *
   * \return The transformed module.
   */
  Module operator()(const Module& mod,
                    const PassContext& pass_ctx) const {
    const PassNode* node = operator->();
    CHECK(node != nullptr);
    return node->operator()(mod, pass_ctx);
  }

  TVM_DEFINE_NODE_REF_METHODS(Pass, NodeRef, PassNode);
};

class SequentialNode;

class Sequential : public Pass {
 public:
  /*!
   * \brief The constructor of `Sequential`.
   *
   * \param passes The passes to apply.
   * \param pass_info The pass metadata.
   */
  TVM_DLL Sequential(tvm::Array<Pass> passes, PassInfo pass_info);

  /*!
   * \brief The constructor of `Sequential`.
   *
   * \param passes The passes to apply.
   * \param name The name of a sequential pass. It's defaulted to "sequential".
   *        This allows users to only provide a list of passes and execute them
   *        under a given context.
   */
  TVM_DLL Sequential(tvm::Array<Pass> passes, std::string name = "sequential");

  Sequential() = default;
  explicit Sequential(tvm::NodePtr<::tvm::Node> n) : Pass(n) {}

  const SequentialNode* operator->() const;
  using ContainerType = Sequential;
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
TVM_DLL Pass CreateFunctionPass(const runtime::TypedPackedFunc<
                                Function(Function, Module, PassContext)>& pass_func,
                                int opt_level,
                                const std::string& name,
                                const tvm::Array<tvm::Expr>& required);

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
 * \param inline_once whether or not to inline binding used one.
 *
 * \return the pass.
 */
TVM_DLL Pass DeadCodeElimination(bool inline_once = false);

/*!
 * \brief Fold constant expressions.
 *
 * \return The pass.
 */
TVM_DLL Pass FoldConstant();

/*!
 * \brief Fuse operations into expr into seperate functions.
 *
 * \param fuse_opt_level Optimization level. If it is -1 it will be inferred from pass context.
 *
 * \return The pass.
 */
TVM_DLL Pass FuseOps(int fuse_opt_level = -1);

/*!
 * \brief Rewrite the annotated program.
 *
 * \param fallback_device The fallback device which is the default device for
 *                        operators without annotation.
 *
 * \return The pass.
 */
TVM_DLL Pass RewriteAnnotatedOps(int fallback_device);

/*!
 * \brief turn a dataflow graph into Administrative Normal Form, or A-Normal Form (ANF).
 *
 * It will turn an expression that is in a graph form (with sharing implicit),
 * to an expression with explicit sharing (A-Normal Form).
 *
 * The scope of the root expression is the global scope.
 *
 * The scope of any non root expression is the least common ancestor of all it's scope.
 *
 * Values are ordered by post-DFS order in each scope.
 *
 * \return The pass.
 */
TVM_DLL Pass ToANormalForm();

/*!
 * \brief Turn an expression into continuation passing style(CPS).
 *
 * CPS mean that every function will, instead of returning the result directly,
 * be passed down an extra function (called the continuation) as argument,
 * and pass the result to the continuation instead.
 *
 * Thus, every function call has to be passed an extra argument
 * that represent the rest of the computation (Hence the name of continuation).
 *
 * Similarly, all other compute will be wrapped and call the continuation as well.
 *
 * \return the pass.
 */
TVM_DLL Pass ToCPS();

/*!
 * \brief Remove let binding and directly share via pointer instead.
 *
 * It will remove all let binding,
 * and turn all of the variable bound by let into direct pointer reference.
 *
 * \return the expression in graph normal form.
 */
TVM_DLL Pass ToGraphNormalForm();

/*!
 * \brief Aggressive constant propagation/constant folding/inlining.
 *
 * It will do as much computation in compile time as possible.
 * It has two benefit: remove runtime overhead, and allow more optimization (typically fusion).
 * As a side effect, code size will explode.
 *
 * \return the optimized expression.
 */
TVM_DLL Pass PartialEval();

/*!
 * \brief Simplify certain operators during inference. For example, batch norm
 * will be unpacked into a number of simplified operators.
 *
 * \return The Pass.
 */
TVM_DLL Pass SimplifyInference();

/*!
 * \brief Infer the type of an expression.
 *
 * The result of type checking is a new expression with unambigous
 * type information filled in, as well as it's checked type field
 * populated with the result type.
 *
 * \return The pass.
 */
TVM_DLL Pass InferType();

/*!
 * \brief Search and eliminate common subexpression. For example, if there are
 * two expressions evaluated to an identical value, a single variable is created
 * and these two expressions are replaced by this variable.
 *
 * \param fskip The callback argument that allows to skip certain expressions.
 *
 * \return The pass.
 */
TVM_DLL Pass EliminateCommonSubexpr(PackedFunc fskip = nullptr);

/*!
 * \brief Combine parallel 2d convolutions into a single convolution if the
 * number of branches of this conv2d operator is not less than
 * `min_num_branch`.
 *
 * \param min_num_branches The minimun number of branches.
 *
 * \return The pass.
 */
TVM_DLL Pass CombineParallelConv2D(uint64_t min_num_branches = 3);

/*!
 * \brief Backward fold axis scaling into weights of conv/dense operators.
 *
 * \return The pass.
 */
TVM_DLL Pass BackwardFoldScaleAxis();

/*!
 * \brief Forward fold axis scaling into weights of conv/dense operators.
 *
 * \return The pass.
 */
TVM_DLL Pass ForwardFoldScaleAxis();

/*!
 * \brief A sequential pass that executes ForwardFoldScaleAxis and
 * BackwardFoldScaleAxis passes.
 *
 * \return The pass.
 */
TVM_DLL Pass FoldScaleAxis();

/*!
 * \brief Canonicalize some operators to the simplified operators. For example,
 * bias_add can be canonicalized to expand_dims and broadcast_add.
 *
 * \return The pass.
 */
TVM_DLL Pass CanonicalizeOps();

/*!
 * \brief Alternate the layouts of operators or replace primitive operators
 * with other expressions.
 *
 * \return The pass.
 */
TVM_DLL Pass AlterOpLayout();

/*!
 * \brief Legalizes an expr with another expression.
 * \param legalize_map_attr_name The Op's attr name which corresponds to the legalize rule function.
 * One can collect and isolate similar type of legalize transformations using this param. For
 * example, transformations that only apply to Dialects can be isolated into a FTVMDialectLegalize
 * string. This pass calls only those transformations that have been registered using the supplied
 * legalize_map_attr_name.
 *
 * \return The pass.
 */
TVM_DLL Pass Legalize(const std::string& legalize_map_attr_name = "FTVMLegalize");

/*!
 * \brief Canonicalize cast expressions to make operator fusion more efficient.
 *
 * \return The pass.
 */
TVM_DLL Pass CanonicalizeCast();

/*!
 * \brief Add abstraction over a function
 *
 * For example: `square` is transformed to
 * `fun x -> square x`.
 *
 * See https://en.wikipedia.org/wiki/Lambda_calculus#%CE%B7-conversion
 * for more details.
 *
 * \return The pass.
 */
TVM_DLL Pass EtaExpand();

/*!
 * \brief Print the IR for a module to help debugging.
 *
 * \return the pass.
 */
TVM_DLL Pass PrintIR();

}  // namespace transform

/*!
 * \brief Bind the free variables to a Relay expression. This is a helper
 * function usually called by other pass functions to help optimizations.
 *
 * \param expr The input expression.
 * \param binds The variable to expression map that will be used to help the
 *        binding.
 *
 * \return The updated expression.
 */
TVM_DLL Expr Bind(const Expr& expr, const tvm::Map<Var, Expr>& binds);

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
TVM_DLL Function InferType(const Function& f,
                           const Module& mod,
                           const GlobalVar& var);

/*!
 * \brief Apply rewrite rules to rewrite the expr in post DFS order. This
 * function is used as a helper function to rewrtie an expression in a pass.
 *
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
 * \brief Apply rewrite rules to rewrite the expr in post DFS order. This
 * function is used as a helper function to rewrtie an expression in a pass.
 *
 * \param expr The expression.
 * \param rewrite_func The rewrite func that will apply to all operators.
 * \param fcontext Additional callback to provide context argument for each call node.
 * \param fmulti_ref_trigger Transformation function to be called when
 *                           an Expr consumed by multiple callers.
 *
 * \return The rewritten expression.
 */
TVM_DLL Expr ForwardRewrite(const Expr& expr,
                            const FForwardRewrite& rewrite_func,
                            std::function<NodeRef(const Call&)> fcontext = nullptr,
                            std::function<Expr(const Expr&)> fmulti_ref_trigger = nullptr);

/*!
 * \brief Rewrite the annotated program.
 *
 * \param expr The expression.
 * \param fallback_device The fallback device which is the default device for
 *                        operators without annotation.
 *
 * \return The updated program.
 */
TVM_DLL Expr RewriteAnnotatedOps(const Expr& expr, int fallback_device);

/*!
 * \brief Turn an expression into continuation passing style(CPS).
 *
 * CPS mean that every function will, instead of returning the result directly,
 * be passed down an extra function (called the continuation) as argument,
 * and pass the result to the continuation instead.
 *
 * Thus, every function call has to be passed an extra argument
 * that represent the rest of the computation (Hence the name of continuation).
 *
 * Similarly, all other compute will be wrapped and call the continuation as well.
 *
 * \param f the function.
 * \param mod the module.
 *
 * \return the converted Function.
 */
TVM_DLL Function ToCPS(const Function& f, const Module& mod);

/*!
 * \brief Remove the continuation argument of a CPS function.
 *
 * Note that this only transform the type back into un-CPS form
 * when there is no higher order input/output.
 *
 * \param f the function.
 *
 * \return the converted Function.
 */
TVM_DLL Function UnCPS(const Function& f);

/*!
 * \brief Deduplicate the bound variables and type variables in the expression.
 *
 * \param e the expression.
 *
 * \return the deduplicated expression.
 */
TVM_DLL Expr DeDup(const Expr& e);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TRANSFORM_H_
