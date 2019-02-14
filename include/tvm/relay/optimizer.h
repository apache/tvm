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
#include <tvm/ir.h>
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
  kFunctionKind = 2
};

class PassContext;

/*
 * \brief PassContextNode contains the information that a pass can rely on, such as
 * the analysis result.
 */
class PassContextNode : public RelayNode {
 public:
  /*!
   * \brief The error reporter used to notify users why an optimization fails.
   */
  ErrorReporter err_reporter_;

  PassContextNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
  }

  TVM_DLL static PassContext make();

  static constexpr const char* _type_key = "relay.PassContext";
  TVM_DECLARE_NODE_TYPE_INFO(PassContextNode, RelayNode);
};

class PassContext : public NodeRef {
 public:
  PassContext() = default;
  explicit PassContext(NodePtr<tvm::Node> p) : NodeRef(p) {}

  const PassContextNode* operator->() const {
    return static_cast<PassContextNode*>(this->node_.get());
  }

  using ContainerType = PassContextNode;
};

// We use currying here. The Relay module is captured for optimizations. It
// runs on a Relay node type NodeT and yields a new node with the same type.
// For example, PassFunc<Function> indicates we perform a Function to Function
// transformation on the given Module.
template <typename NodeT,
          typename = std::enable_if<(std::is_same<NodeT, Module>::value ||
                                     std::is_same<NodeT, Function>::value)>>
using PassFunc =
    runtime::TypedPackedFunc<runtime::TypedPackedFunc<NodeT(NodeT)>(
        const Module& mod)>;

class Pass;

/*!
 * \brief PassNode is the base type of differnt types of optimization passes.
 * It is implemented by different pass subclasses at different granularity of
 * Relay nodes.
 */
class PassNode : public RelayNode {
 public:
  /*! \brief The name of an optimization/analysis pass. */
  std::string name;
  /*! \brief The minimal optimization level that this pass will be enabled. */
  int opt_level;
  /*! \brief The kind of an optimization/analysis pass. */
  PassKind pass_kind;
  /*! \brief The flag to indicate if a pass is enabled. */
  bool enabled;
  /*!
   * \brief The passes that are required by this pass.
   * TODO(zhiics) required_passes are used to identify the dependency of
   * different passes. We will use it build the pass dependency graph in the
   * followup PRs.
   */
  tvm::Array<tvm::Expr> required_passes;

  /*!
   * \brief Get the required passes for this pass as a vector of std::string.
   */
  TVM_DLL std::vector<std::string> RequiredPasses() const;

  /*!
   * \brief Execute the optimization pass using a functor. This functor invokes
   * the `run` method to perform a real optimization on a certain type of node.
   *
   * \param mod The module that an optimization pass runs on.
   * \param pass_ctx The context information that is used to help perform
   *        a given pass.
   */
  void operator()(Module* mod, const PassContext& pass_ctx) {
    Run(mod, pass_ctx);
  }

  /*!
   * \brief Execute the optimization pass. This is function should be specilized
   * for different types of Relay nodes. For example, we mainly allow
   * transformation of from Module/Function to Module/Function. Note  that the
   * module will be updated.
   *
   * \param mod The module that an optimization pass runs on.
   * \param pass_ctx The context information that is used to help perform
   *        a given pass.
   *
   * \return Return the updated module through mod.
   */
  virtual void Run(Module* mod, const PassContext& pass_ctx) const = 0;

  void VisitAttrs(tvm::AttrVisitor* v) override {
    v->Visit("name", &name);
    v->Visit("opt_level", &opt_level);
    v->Visit("passkind", &pass_kind);
    v->Visit("enabled", &enabled);
    v->Visit("required_passes", &required_passes);
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
    v->Visit("enabled", &enabled);
    v->Visit("required_passes", &required_passes);
  }

  /*!
   * \brief Run a function pass on a certain module.
   *
   * \param mod The module that an optimization pass runs on.
   * \param pass_ctx The context information that is used to help perform
   *        a given module pass.
   *
   * \return Return the updated module through mod.
   */
  void Run(Module* mod, const PassContext& pass_ctx) const override;

  TVM_DLL static ModulePass make(std::string name, int opt_level,
                                 PassFunc<Module> pass_func,
                                 bool enabled = true,
                                 tvm::Array<tvm::Expr> required_passes = {});

  static constexpr const char* _type_key = "relay.ModulePass";
  TVM_DECLARE_NODE_TYPE_INFO(ModulePassNode, PassNode);
};

RELAY_DEFINE_NODE_REF(ModulePass, ModulePassNode, Pass);

class FunctionPass;

/*!
 * \brief Function-level passes are used to implement various global
 * optimizations for a given Relay module. It fetches one function at a time
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
    v->Visit("enabled", &enabled);
    v->Visit("required_passes", &required_passes);
  }

  /*!
   * \brief Run a function pass on a certain module.
   *
   * \param mod The module that an optimization pass runs on.
   * \param pass_ctx The context information that is used to help perform
   *        a given pass.
   *
   * \return Return the updated module through mod.
   */
  void Run(Module* mod, const PassContext& pass_ctx) const override;

  TVM_DLL static FunctionPass make(std::string name, int opt_level,
                                   PassFunc<Function> pass_func,
                                   bool enabled = true,
                                   tvm::Array<tvm::Expr> required_passes = {});

  static constexpr const char* _type_key = "relay.FunctionPass";
  TVM_DECLARE_NODE_TYPE_INFO(FunctionPassNode, PassNode);

 protected:
  bool SkipFunction(const Function& func) const;
};

RELAY_DEFINE_NODE_REF(FunctionPass, FunctionPassNode, Pass);

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
 * TODO(jroesch, zhiics): We are currently using a very simple design for the
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
  Optimizer(const Module& mod, const tvm::Array<Pass>& passes,
            const PassContext& pass_ctx)
      : module_(mod), passes_(passes), pass_ctx_(pass_ctx) {}

  /*!
   * \brief Add a pass to the pass list.
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
  /* \brief The module where a host of passes are executed on. It is designed
   * to be mutable because each optimization is likely to update the module
   * on its completion.
   */
  mutable Module module_;
  /* \brief The pass candidates for optimizations. */
  tvm::Array<Pass> passes_;
  /* \brief The auxiliary pass context/information that is used to help perform
   * the given list of passes.*/
  PassContext pass_ctx_;
  friend void Optimize(const tvm::Array<Pass>& passes,
                       Module* mod,
                       const PassContext& pass_ctx);
};

/*!
 * \brief Optimizes the functions and/or expressions in the module. This free
 * function is designed as a template function that could take different types
 * of Relay nodes.

 * \param passes The optimization passes.
 * \param mod The module where optimizations are performed on.
 *        Note that the updated module will be stored and returned.
 * \param pass_ctx The auxiliary pass context/information that is used to help
 *        perform the provided passes.
 *
 * \return Return the updated Module through mod.
 */
void Optimize(const tvm::Array<Pass>& passes,
              Module* mod,
              const PassContext& pass_ctx);

}  // namespace optimize
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OPTIMIZER_H_
