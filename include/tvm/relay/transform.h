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

#include <tvm/packed_func_ext.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/module.h>
#include <string>
#include <vector>

namespace tvm {
namespace relay {
namespace transform {

/*!
 * \brief A data structure to map the names of specific optimizations to
 *        numeric optimization levels
 */
struct OptPassLevel {
  static const std::unordered_map<std::string, int> CreateMap() {
    const std::unordered_map<std::string, int> m = {
      {"SimplifyInference", 0},
      {"OpFusion", 1},
      {"FoldConstant", 2},
      {"CombineParallelConv2D", 3},
      {"FoldScaleAxis", 3},
      {"AlterOpLayout", 3},
      {"CanonicalizeOps", 3},
      {"EliminateCommonSubexpr", 3}
    };
    return m;
  }
  /*!
   * \brief Get level for an optimization pass
   *
   * \param key pass name
   * \return int level
   */
  int operator[](const std::string& key) const {
    const auto data = CreateMap();
    auto it = data.find(key);
    if (it == data.end()) {
      return -1;
    }
    return it->second;
  }
};

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

  /*! \brief The default optimization level. */
  int opt_level{2};

  /*! \brief CPU is the default fallback device for heterogeneous execution. */
  int fallback_device{static_cast<int>(kDLCPU)};

  /*! \brief The list of required passes. */
  tvm::Array<tvm::Expr> required_pass;
  /*! \brief The list of disabled passes. */
  tvm::Array<tvm::Expr> disabled_pass;

  /*! 
   * \brief A helper struct to get the optimization pass name to opt level
   * mapping.
   */
  OptPassLevel OPT_PASS_LEVEL;

  /*!
   * \brief Convert a list of tvm StringImm to a `std::string` set.
   *
   * \param input. The input StringImm array.
   *
   * \return The coverted `std::strin`g set.
   */
  std::unordered_set<std::string> ToStringSet(
      const tvm::Array<tvm::Expr>& input) const;

  /*!
   * \brief Check if a pass is enabled.
   *
   * \param pass_name The name of an optimization/analysis pass.
   *
   * \return true if the pass is enabled. Otherwise, false.
   */
  bool pass_enabled(const std::string& pass_name) const;

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

class PassContext : public NodeRef {
 public:
  PassContext() {}
  explicit PassContext(tvm::NodePtr<Node> n) : NodeRef(n) {}

  TVM_DLL PassContext(int opt_level, int fallback_device,
                      tvm::Array<tvm::Expr> required_pass,
                      tvm::Array<tvm::Expr> disabled_pass);

  // The entry of a pass context scope.
  TVM_DLL static void EnterWithScope(const PassContext& pass_ctx);
  // The exit of a pass context scope.
  TVM_DLL static void ExitWithScope();
  // Get the currently used pass context.
  TVM_DLL static PassContext Current();

  const PassContextNode* operator->() const;

  using ContainerType = PassContextNode;
  class Internal;

 private:
  // Classes to get the Python `with` like syntax. Enabled after #3231 is merged
  // friend class Internal;
  // friend class With<PassContext>;
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
   * \brief Execute the optimization pass using a functor.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return The updated module.
   */
  virtual Module operator()(const Module& mod) const = 0;

  virtual Module Apply(const Module& mod,
                       const PassContext& pass_ctx) const = 0;

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

class SequentialNode;

class Sequential : public Pass {
 public:
  /*!
   * \brief The constructor of `Sequential`.
   * \param passes The passes to apply.
   * \param pass_info The pass metadata.
   */
  TVM_DLL Sequential(tvm::Array<Pass> passes,
                     PassInfo pass_info);
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
Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, PassContext)>& pass_func,
    int opt_level,
    const std::string& name,
    const tvm::Array<tvm::Expr>& required);

}  // namespace transform
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TRANSFORM_H_
