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
 * \file tvm/ir/transform.h
 *
 * This file implements a pass manager. The pass manager manages a sequence
 * of IRModule -> IRModule transformation passes over a particlar unit of AST. The
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
 * manager performs the IRModule -> IRModule transformation. All
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
#ifndef TVM_IR_TRANSFORM_H_
#define TVM_IR_TRANSFORM_H_

#include <tvm/support/with.h>
#include <tvm/runtime/container.h>
#include <tvm/node/container.h>
#include <tvm/ir/error.h>
#include <tvm/ir/module.h>
#include <string>
#include <utility>

namespace tvm {
namespace transform {

// Forward declare for TraceFunc.
class PassInfo;

/*! \brief A callback for tracing passes, useful for debugging and logging.
 *
 */
using TraceFunc =
  runtime::TypedPackedFunc<void(const IRModule& ir_module,
                                const PassInfo& ctx,
                                bool is_before)>;

/*!
 * \brief PassContextNode contains the information that a pass can rely on,
 * such as analysis results.
 * \sa PassContext
 */
class PassContextNode : public Object {
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
  Array<runtime::String> required_pass;
  /*! \brief The list of disabled passes. */
  Array<runtime::String> disabled_pass;

  TraceFunc trace_func;

  PassContextNode() = default;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("opt_level", &opt_level);
    v->Visit("fallback_device", &fallback_device);
    v->Visit("required_pass", &required_pass);
    v->Visit("disabled_pass", &disabled_pass);
  }

  static constexpr const char* _type_key = "transform.PassContext";
  static constexpr bool _type_has_method_sequal_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(PassContextNode, Object);
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
 * \sa PassContextNode
 */
class PassContext : public ObjectRef {
 public:
  PassContext() {}
  explicit PassContext(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief const accessor.
   * \return const access pointer.
   */
  const PassContextNode* operator->() const {
    CHECK(get() != nullptr);
    return static_cast<const PassContextNode*>(get());
  }
  /*!
   * \brief mutable accessor.
   * \return mutable access pointer.
   */
  PassContextNode* operator->() {
    CHECK(get() != nullptr);
    return static_cast<PassContextNode*>(get_mutable());
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

  /*!
   * \brief Apply the tracing functions of the context to the module, with the info.
   * \param module The IRModule to trace.
   * \param info The pass information.
   * \param is_before Indicated whether the tracing is before or after a pass.
   */
  TVM_DLL void Trace(const IRModule& module, const PassInfo& info, bool is_before) const;

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
  friend class With<PassContext>;
};

/*!
 * \brief Meta data that will be used to help optimization and analysis.
 * \sa PassInfo
 */
class PassInfoNode : public Object {
 public:
  /*! \brief The minimal optimization level that this pass will be enabled. */
  int opt_level;

  /*! \brief The name of an optimization/analysis pass. */
  std::string name;

  /*! \brief The passes that are required to perform the current pass. */
  Array<runtime::String> required;

  PassInfoNode() = default;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("opt_level", &opt_level);
    v->Visit("name", &name);
    v->Visit("required", &required);
  }

  static constexpr const char* _type_key = "transform.PassInfo";
  static constexpr bool _type_has_method_sequal_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(PassInfoNode, Object);
};

/*
 * \brief Managed reference class for PassInfoNode
 * \sa PassInfoNode
 */
class PassInfo : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param opt_level The optimization level
   * \param name Name of the pass.
   * \param required  The passes that are required to perform the current pass.
   */
  TVM_DLL PassInfo(int opt_level,
                   std::string name,
                   Array<runtime::String> required);

  TVM_DEFINE_OBJECT_REF_METHODS(PassInfo, ObjectRef, PassInfoNode);
};

/*!
 * \brief PassNode is the base type of differnt types of optimization passes.
 * It is designed as a pure class and implemented by different pass subclasses
 * at different granularity of Relay nodes.
 */
class PassNode : public Object {
 public:
  virtual ~PassNode() {}
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
  IRModule operator()(IRModule mod) const {
    return this->operator()(std::move(mod), PassContext::Current());
  }

  /*!
   * \brief Transform mod using a functor under a given pass context.
   *
   * \param mod The module that an optimization pass runs on.
   * \param pass_ctx The pass context that can provide information for the optimization.
   *
   * \return The transformed module.
   */
  virtual IRModule operator()(IRModule mod,
                              const PassContext& pass_ctx) const = 0;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "transform.Pass";
  TVM_DECLARE_BASE_OBJECT_INFO(PassNode, Object);
};

class Pass : public ObjectRef {
 public:
  /*!
   * \brief Transform mod using the default PassContext in the current scope.
   *
   * \code
   *
   * // If you do no longer need the input module
   * // it is recommended to use std::move to move your input module.
   * mod = pass(std::move(mod));
   *
   * \endcode
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return The transformed module.
   */
  IRModule operator()(IRModule mod) const {
    const PassNode* node = operator->();
    CHECK(node != nullptr);
    return node->operator()(std::move(mod));
  }
  /*!
   * \brief Transform mod using a functor under a given pass context.
   *
   * \param mod The module that an optimization pass runs on.
   * \param pass_ctx The pass context that can provide information for the optimization.
   *
   * \return The transformed module.
   */
  IRModule operator()(IRModule mod,
                      const PassContext& pass_ctx) const {
    const PassNode* node = operator->();
    CHECK(node != nullptr);
    return node->operator()(std::move(mod), pass_ctx);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(Pass, ObjectRef, PassNode);
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
  TVM_DLL Sequential(Array<Pass> passes, PassInfo pass_info);

  /*!
   * \brief The constructor of `Sequential`.
   *
   * \param passes The passes to apply.
   * \param name The name of a sequential pass. It's defaulted to "sequential".
   *        This allows users to only provide a list of passes and execute them
   *        under a given context.
   */
  TVM_DLL Sequential(Array<Pass> passes, std::string name = "sequential");

  Sequential() = default;
  explicit Sequential(ObjectPtr<Object> n) : Pass(n) {}

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
TVM_DLL Pass CreateModulePass(
    const runtime::TypedPackedFunc<IRModule(IRModule, PassContext)>& pass_func,
    int opt_level,
    const std::string& name,
    const Array<runtime::String>& required);


/*!
 * \brief A special trace pass that prints the header and IR to LOG(INFO).
 * \param header The header to be attached to the output.
 * \param show_meta_data Whether should we show meta data.
 * \return The pass.
 */
TVM_DLL Pass PrintIR(std::string header = "", bool show_meta_data = false);

}  // namespace transform
}  // namespace tvm

#endif  // TVM_IR_TRANSFORM_H_
