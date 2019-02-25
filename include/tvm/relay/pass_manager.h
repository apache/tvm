/*!  * Copyright (c) 2019 by Contributors
 * \file tvm/relay/pass_manager.h
 *
 * \brief The pass manager manages a sequence of Relay-to-Relay transformation
 * passes over a particlar unit of AST. The design is largely inspired from
 * LLVM's pass manager.
 *
 * The responsibilities of pass managers usually at least involve:
 *  - organizing the execution orders of optimization passes though not
 * necessarily in the optimal sequence.
 *  - collecting required analysis information and keep them up-to-date before
 * pass to run.
 *  - simplifying the implementation of new passes for compiler developers, etc.
 *
 * TODO(jroesch, zhiics): We are currently using a very simple design for the
 * pass manager, i.e. it just execute on certain pass or a list of passes that
 * run in order.
 *
 * As we move forward we need to generalize the ability to have constraints
 * between them. For example, we might need to preserve the dependencies between
 * different passes and validate them on the completion of a certain pass.
 *
 * We also need to store side information and import the error reporting system.
 */
#ifndef TVM_RELAY_PASS_MANAGER_H_
#define TVM_RELAY_PASS_MANAGER_H_

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
namespace pass {

class PassContext;

/*!
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

// We use currying here. It runs on a Relay node type NodeT and yields a new
// node with the same type. The Relay module is captured for optimizations as
// most of the current Relay optimizations are module to module. Currying
// sketches the optimization, i.e. how we want to mutate an AST, and it is
// passed as packed functions that will be invoked when called by `run`.
//
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

  /*! \brief Set the context information for a pass. */
  void SetContext(const PassContext& pass_ctx) {
    pass_ctx_ = pass_ctx;
  }

  /*!
   * \brief Get the required passes for this pass as a vector of std::string.
   */
  virtual std::vector<std::string> Required() const = 0;

  /*!
   * \brief Execute the optimization pass using a functor. This functor invokes
   *        the `run` method to perform a real optimization on a certain type
   *        of node.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return The updated module.
   */
  virtual Module operator()(const Module& mod) const = 0;

  /*!
   * \brief Execute the optimization pass. This is function should be specilized
   *        for different types of Relay nodes. For example, we mainly allow
   *        transformation of from Module/Function to Module/Function. Note that
   *        the module will be updated.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return Return the updated module.
   */
  virtual Module Run(const Module& mod) const = 0;

  void VisitAttrs(tvm::AttrVisitor* v) override {
    v->Visit("name", &name);
    v->Visit("opt_level", &opt_level);
  }

  static constexpr const char* _type_key = "relay.Pass";
  TVM_DECLARE_BASE_NODE_INFO(PassNode, RelayNode);

 protected:
  /*!
   * \brief The context information that is used to help perform a given pass.
   */
  PassContext pass_ctx_;
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

class ModulePass;

/*!
 * \brief Module-level passes are designed to implement global
 * analysis/optimizations, i.e. interprocedural optimizations (IPO), etc. Passes
 * at this level have the full control of a given Relay program including
 * addition and deletion of functions.
 */
class ModulePassNode : public PassNode {
 public:
  /*! \brief The curried function sketches the real optimization. */
  PassFunc<Module> pass_func;

  ModulePassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) override {
    v->Visit("name", &name);
    v->Visit("opt_level", &opt_level);
  }

  Module operator()(const Module& mod) const override {
    return Run(mod);
  }

  /*!
   * \brief Run a function pass on a certain module.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return Return the updated module.
   */
  Module Run(const Module& mod) const override;

  /*! \brief Collect the required passes for this module pass. */
  std::vector<std::string> Required() const override;

  TVM_DLL static ModulePass make(std::string name, int opt_level,
                                 PassFunc<Module> pass_func);

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
  /*! \brief The curried packed function that sketches the real optimization. */
  PassFunc<Function> pass_func;

  FunctionPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) override {
    v->Visit("name", &name);
    v->Visit("opt_level", &opt_level);
  }

  Module operator()(const Module& mod) const override {
    return Run(mod);
  }

  /*!
   * \brief Run a function pass on a certain module.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return Return the updated module.
   */
  Module Run(const Module& mod) const override;

  /*! \brief Collect the required passes for this module pass. */
  std::vector<std::string> Required() const override;

  TVM_DLL static FunctionPass make(std::string name, int opt_level,
                                   PassFunc<Function> pass_func);

  static constexpr const char* _type_key = "relay.FunctionPass";
  TVM_DECLARE_NODE_TYPE_INFO(FunctionPassNode, PassNode);

 protected:
  bool SkipFunction(const Function& func) const;
};

RELAY_DEFINE_NODE_REF(FunctionPass, FunctionPassNode, Pass);

class SequentialPass;

/*!
 * \brief The SequentialPassNode contains a set of passes that transform Relay
 * programs from one AST to another semantically equivalent one.
 *
 * One example of this level of pass is that the pass manager needs to correctly
 * perform a host of optimizations with a given optimization level and disabled
 * passes.
 */
class SequentialPassNode : public PassNode {
 public:
  /*! \brief A list of passes that used to compose a sequential pass. */
  tvm::Array<Pass> passes;
  /*!
   * \brief A list of disabled passes that should be excluded when executing the
   * sequential pass.
   */
  tvm::Array<tvm::Expr> disabled;

  void VisitAttrs(tvm::AttrVisitor* v) override {
    v->Visit("name", &name);
    v->Visit("opt_level", &opt_level);
    v->Visit("passes", &passes);
    v->Visit("disabled", &disabled);
  }

  /*!
   * \brief Add a pass to the pass list.
   *
   * \param pass The candidate pass to be added.
   */
  void AddPass(const Pass& pass) {
    passes.push_back(pass);
  }

  TVM_DLL static SequentialPass make(std::string name, int opt_level,
                                     tvm::Array<Pass> passes,
                                     tvm::Array<tvm::Expr> disabled);

  /*!
   * \brief Resolve the pass dependency. It globs all required passes by
   *        a given pass and executes them.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return The updated module after resolving pass dependencies.
   *
   * TODO(zhiics) Build a dependency graph among the passes using provided
   * metadata, i.e. required_passes. Likely, we can have a data structure, i.e.
   * PassInfo, to store the relevant information including the parent passes.
   */
  void ResolveDependency(const Module& mod);

  std::vector<std::string> Required() const override;

  Module operator()(const Module& mod) const override {
    return Run(mod);
  }

  TVM_DLL std::vector<std::string> DisabledPasses() const;

  /*!
   * \brief Perform optimizations on a series of passes. The aforementioned
   *        typical pass manager jobs could be done by it. This function could
   *        be overloaded to focus on different metrics, i.e. performance,
   *        memory footprint, etc.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return Return the updated module.
   */
  Module Run(const Module& mod) const override;

  static constexpr const char* _type_key = "relay.SequentialPass";
  TVM_DECLARE_NODE_TYPE_INFO(SequentialPassNode, PassNode);
};

RELAY_DEFINE_NODE_REF(SequentialPass, SequentialPassNode, Pass);

/*
 * \brief Create a module pass.
 *
 * \param name The name of the module pass.
 * \param opt_level The optimization level of the module pass.
 * \param pass_func The curried packed function that contains the optimization.
 *
 * \return The created module pass.
 */
ModulePass CreateModulePass(const std::string& name, int opt_level,
                            const PassFunc<Module>& pass_func);

/*
 * \brief Create a function pass.
 *
 * \param name The name of the function pass.
 * \param opt_level The optimization level of the function pass.
 * \param pass_func The curried packed function that contains the optimization.
 *
 * \return The created function pass.
 */
FunctionPass CreateFunctionPass(const std::string& name, int opt_level,
                                const PassFunc<Function>& pass_func);
/*
 * \brief Create a sequential pass.
 *
 * \param name The name of the sequential pass.
 * \param opt_level The optimization level of the sequential pass. It could be
 *        the highest opt_level of the list of passes.
 * \param passes The optimization passes will be performed.
 * \param disabled The disabled passes.
 *
 * \return The created sequential pass.
 */
SequentialPass CreateSequentialPass(const std::string& name, int opt_level,
                                    const tvm::Array<Pass>& passes,
                                    const tvm::Array<tvm::Expr>& disabled);

}  // namespace pass
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PASS_MANAGER_H_
