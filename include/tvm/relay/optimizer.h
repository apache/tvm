/*!
 * Copyright (c) 2019 by Contributors
 * \file tvm/relay/optimizer.h
 *
 * \brief The optimizer manages a sequence of Relay-to-Relay transformation
 * passes over a particlar unit of AST. The design is largely inspired from
 * LLVM's pass manager.
 */
#ifndef TVM_RELAY_OPTIMIZER_H_
#define TVM_RELAY_OPTIMIZER_H_

#include <tvm/attrs.h>
#include <tvm/packed_func_ext.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/module.h>
#include <tvm/relay/type.h>

#include <string>
#include <vector>

namespace tvm {
namespace relay {
namespace optimize {

/*! \brief An enumerator to represent the granularity of different passes. */
enum PassKind : int {
  kModuleKind = 1,
  kFunctionKind = 2,
  kExprKind = 3
};

class PassState;

class PassStateNode : public RelayNode {
 public:
  /*! \brief The module that a pass state contains. */
  Module mod;
  /*! \brief The error reporter used to notify users why an optimization fails.
   */
  ErrorReporter err_reporter;

  PassStateNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("mod", &mod);
  }

  TVM_DLL static PassState make(Module mod);

  static constexpr const char* _type_key = "relay.PassState";
  TVM_DECLARE_NODE_TYPE_INFO(PassStateNode, RelayNode);
};

class PassState : public NodeRef {
 public:
  PassState() = default;
  explicit PassState(NodePtr<tvm::Node> p) : NodeRef(p) {}

  const PassStateNode* operator->() const {
    return static_cast<PassStateNode*>(this->node_.get());
  }

  using ContainerType = PassStateNode;
};

// We use currying here. The pass state is captured as for optimizations. It
// produces a function from T to R. For example, PassFunc<Function, Function>
// indicates we perform Function to Function transformation on the captured
// Module.
template <
    typename T, typename R = T,
    typename = std::enable_if<
        (std::is_same<T, Module>::value || std::is_same<T, Function>::value ||
         std::is_same<T, Expr>::value) &&
        (std::is_same<R, Module>::value || std::is_same<R, Function>::value ||
         std::is_same<R, Expr>::value)>>
using PassFunc = runtime::TypedPackedFunc<runtime::TypedPackedFunc<R(T)>(
    const PassState& state)>;

class Pass;

/*!
 * \brief PassNode is the base type of the Relay type hierarchy. It is
 * implemented by different pass subclasses.
 */
class PassNode : public RelayNode {
 public:
  /*! \brief The name of an optimization/analysis pass. */
  std::string name;
  /*! \brief The minimal optimization level that this pass will be enabled. */
  int opt_level;
  /*! \brief The kind of an optimization/analysis pass. */
  PassKind pass_kind;

  /*!
   * \brief Execute the optimization pass. This is function should be specilized
   * for different types of AST nodes. For example, we mainly allow
   * transformation of from Module/Function/Expr to Module/Function/Expr.
   *
   * \param state The pass state that an optimization pass runs on.
   */
  virtual void run(PassState* state) const = 0;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("opt_level", &opt_level);
    v->Visit("passkind", &pass_kind);
  }

  static constexpr const char* _type_key = "relay.Pass";
  TVM_DECLARE_BASE_NODE_INFO(PassNode, RelayNode);
};

class Pass : public NodeRef {
 public:
  Pass() = default;
  explicit Pass(NodePtr<tvm::Node> p) : NodeRef(p) {}

  const PassNode* operator->() const {
    return static_cast<PassNode*>(this->node_.get());
  }

  using ContainerType = PassNode;
};

class ModulePass;

/*!
 * \brief Module-level passes are designed to implement global
 * analysis/optimizations, i.e. interprocedural optimizations (IPO), etc. Passes
 * at this level have the full control of a given Relay program including
 * addition and deletion of functions.
 */
class ModulePassNode : public PassNode {
 public:
  PassFunc<Module> pass_func;

  ModulePassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("opt_level", &opt_level);
    v->Visit("pass_kind", &pass_kind);
  }

  void run(PassState* state) const override;

  TVM_DLL static ModulePass make(std::string name, int opt_level,
                                 PassFunc<Module> pass_func);

  static constexpr const char* _type_key = "relay.ModulePass";
  TVM_DECLARE_NODE_TYPE_INFO(ModulePassNode, PassNode);
};

RELAY_DEFINE_NODE_REF(ModulePass, ModulePassNode, Pass);

class FunctionPass;

/*!
 * \brief Function-level passes are used to implement various global
 * optimizations for a given pass state. It fetches one function at a time
 * from the function list in the module for optimization.
 *
 * Note that the scope of passes at this level is a Relay function. Therefore,
 * we cannot add or delete a function through these passes as they are not aware
 * of the global information.
 */
class FunctionPassNode : public PassNode {
 public:
  PassFunc<Function> pass_func;

  FunctionPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("opt_level", &opt_level);
    v->Visit("pass_kind", &pass_kind);
  }

  /*
   * !\brief Run a function pass on a certain pass state.
   */
  void run(PassState* state) const override;
  TVM_DLL static FunctionPass make(std::string name, int opt_level,
                                   PassFunc<Function> pass_func);

  static constexpr const char* _type_key = "relay.FunctionPass";
  TVM_DECLARE_NODE_TYPE_INFO(FunctionPassNode, PassNode);

 protected:
  bool SkipFunction(const Function& func) const;
};

RELAY_DEFINE_NODE_REF(FunctionPass, FunctionPassNode, Pass);

class ExprPass;

/*!
 * \brief TODO(zhiics) Should we design expr passes as the basic block passes in
 * LLVM? Basic block passes are mainly focusing on local optimizations.
 */
class ExprPassNode : public PassNode {
 public:
  PassFunc<Expr> pass_func;

  ExprPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("opt_level", &opt_level);
    v->Visit("pass_kind", &pass_kind);
  }

  void run(PassState* state) const override;

  TVM_DLL static ExprPass make(std::string name, int opt_level,
                                  PassFunc<Expr> pass_func);

  static constexpr const char* _type_key = "relay.ExprPass";
  TVM_DECLARE_NODE_TYPE_INFO(ExprPassNode, PassNode);
};

RELAY_DEFINE_NODE_REF(ExprPass, ExprPassNode, Pass);

/*!
 * \brief The Relay pass manager contains a set of passes which transform Relay
 * programs from one ast to another semantically equivalent one. The
 * responsibilities of pass managers usually at least involve:
 *  - organizing the execution orders of optimization passes though not
 * necessarily in the optimal sequence.
 *  - collecting required analysis information and keep them up-to-date before
 * pass to run.
 *  - simplifying the implementation of new passes for compiler developers, etc.
 *
 * TODO(@jroesch, @zhiics): We are currently using a very simple design for the
 * pass manager, i.e. it just stores a list of passes that run in order.
 *
 * As we move forward we need to generalize the ability to have constraints
 * between them. For example, we might need to preserve the dependencies between
 * different passes and validate them on the completion of a certain pass.
 *
 * We also need to store side information and import the error reporting system.
 */
class Optimizer {
 public:
  Optimizer(const PassState& state, const tvm::Array<Pass>& passes)
      : state_(state), passes_(passes) {}  // NOLINT(*)

  /*!
   * \brief Add a pass to the pass array.
   *
   * \param pass The candidate pass to be added.
   */
  void AddPass(const Pass& pass) {
    passes_.push_back(pass);
  }

  // TODO(zhiics) Build a dependency graph among the passes using provided
  // metadata, such as pass name/id. Likely, we can have a data structure, i.e.
  // PassInfo, to store the relevant information including the parent passes.
  void BuildDependencyGraph();

  /*!
   * \brief Perform optimizations on a series of passes. The aforementioned
   * typical pass manager jobs could be done by it. This function could be
   * overloaded to focus on different metrics, i.e. performance, memory
   * footprint, etc.
   */
  void Optimize() const;

 private:
  /* \brief The pass state where a host of passes are executed on. It is
   * designed to be mutable because each optimization is likely to update the
   * state on its completion.
   */
  mutable PassState state_;
  /* \brief The pass candidates for optimizations. */
  tvm::Array<Pass> passes_;
  friend void Optimize(const tvm::Array<Pass>& passes, PassState* state);
};

/*!
 * \brief Optimizes the functions and/or expressions in the pass state. This
 * free function is designed as a template function that could take different
 * types of Relay nodes.

 * \param passes The optimization passes.
 * \param state The pass state for optimization. Note that `state` is mutable.
 *        The updated state will be stored and returned.

 */
void Optimize(const tvm::Array<Pass>& passes, PassState* state);

}  // namespace optimize
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OPTIMIZER_H_
