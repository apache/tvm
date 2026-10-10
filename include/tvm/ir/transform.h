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
 * \brief Utilities for IR transformations.
 *
 * Pass represents an IRModule-to-IRModule transformation.
 * PassContext provides scoped configuration and instrumentation.
 * Sequential applies a list of passes in order.
 */
#ifndef TVM_IR_TRANSFORM_H_
#define TVM_IR_TRANSFORM_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/creator.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/module.h>
#include <tvm/ir/with_context.h>

#include <string>
#include <type_traits>
#include <utility>

namespace tvm {
namespace transform {

class PassInfo;
class PassContextNode;

/*!
 * \brief PassContext that is used to configure the pass behavior.
 *
 * \code
 *
 *  auto new_ctx = PassContext::Create();
 *  ctx->opt_level = 2;
 *  With<PassContext> scope(ctx);
 *  // pass context in effect.
 *
 * \endcode
 * \sa PassContextNode
 */
class PassContext : public ffi::ObjectRef {
 public:
  static constexpr bool _type_is_nullable = false;
  /*!
   * \brief constructor with UnsafeInit
   */
  explicit PassContext(ffi::UnsafeInit tag) : ffi::ObjectRef(tag) {}
  /*!
   * \brief constructor with ffi::ObjectPtr
   */
  explicit PassContext(ffi::ObjectPtr<PassContextNode> n);
  /*!
   * \brief const accessor.
   * \return const access pointer.
   */
  const PassContextNode* operator->() const;
  /*!
   * \brief mutable accessor.
   * \return mutable access pointer.
   */
  PassContextNode* operator->();

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
   * \brief Get all supported configuration names and metadata, registered within the PassContext.
   * \return Map indexed by the config name, pointing to the metadata map as key-value
   */
  TVM_DLL static ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::String>> ListConfigs();

  /*!
   * \brief Call instrument implementations' callbacks when entering PassContext.
   *        The callbacks are called in order, and if one raises an exception, the rest will not be
   *        called.
   */
  TVM_DLL void InstrumentEnterPassContext();

  /*!
   * \brief Call instrument implementations' callbacks when exiting PassContext.
   *        The callbacks are called in order, and if one raises an exception, the rest will not be
   *        called.
   */
  TVM_DLL void InstrumentExitPassContext();

  /*!
   * \brief Call instrument implementations' callbacks before a pass run.
   *        The callbacks are called in order, and if one raises an exception, the rest will not be
   *        called.
   *
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   *
   * \return false: the pass is skipped; true: the pass runs.
   */
  TVM_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;

  /*!
   * \brief Call instrument implementations callbacks after a pass run.
   *        The callbacks are called in order, and if one raises an exception, the rest will not be
   *        called.
   *
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   */
  TVM_DLL void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;

  /*!
   * \brief Check whether a pass is enabled.
   * \param info The pass information.
   * \return true if the pass is enabled. Otherwise, false.
   */
  TVM_DLL bool PassEnabled(const PassInfo& info) const;

  /*!
   * \brief Register a valid configuration option and its ValueType for validation.
   *
   * \param key The configuration key.
   * \tparam ValueType The value type to be registered
   */
  template <typename ValueType>
  static void RegisterConfigOption(const char* key) {
    if constexpr (std::is_base_of_v<ffi::ObjectRef, ValueType>) {
      // Static-init blocks may run before the cached runtime type index is initialized.
      int32_t tindex = ValueType::ContainerType::_GetOrAllocRuntimeTypeIndex();
      auto type_key = ffi::TypeIndexToTypeKey(tindex);
      auto legalization = [=](ffi::Any value) -> ffi::Any {
        if (auto opt_map = value.try_cast<ffi::Map<ffi::String, ffi::Any>>()) {
          return ffi::reflection::ObjectCreator(type_key)(opt_map.value());
        } else {
          auto opt_val = value.try_cast<ValueType>();
          if (!opt_val.has_value()) {
            TVM_FFI_THROW(AttributeError)
                << "Expect config " << key << " to have type " << type_key << ", but instead get "
                << ffi::details::AnyUnsafe::GetMismatchTypeInfo<ValueType>(value);
          }
          return *opt_val;
        }
      };
      RegisterConfigOption(key, type_key, legalization);
    } else {
      // non-object type, do not support implicit conversion from map
      std::string type_str = ffi::TypeTraits<ValueType>::TypeStr();
      auto legalization = [=](ffi::Any value) -> ffi::Any {
        auto opt_val = value.try_cast<ValueType>();
        if (!opt_val.has_value()) {
          TVM_FFI_THROW(AttributeError)
              << "Expect config " << key << " to have type " << type_str << ", but instead get "
              << ffi::details::AnyUnsafe::GetMismatchTypeInfo<ValueType>(value);
        } else {
          return *opt_val;
        }
      };
      RegisterConfigOption(key, type_str, legalization);
    }
  }

  // accessor.
  using ContainerType = PassContextNode;
  class Internal;

 private:
  // The entry of a pass context scope.
  TVM_DLL void EnterWithScope();
  // The exit of a pass context scope.
  TVM_DLL void ExitWithScope();
  // Register configuration key value type.
  TVM_DLL static void RegisterConfigOption(const char* key, ffi::String value_type_str,
                                           std::function<ffi::Any(ffi::Any)> legalization);

  // Classes to get the Python `with` like syntax.
  friend class Internal;
  friend class With<PassContext>;
};

/*!
 * \brief Create a pass-config object with all default values, using the
 *        reflection defaults.
 * \tparam TConfig the ObjectRef type to be created.
 * \return An instance with all reflection-defined default values applied.
 */
template <typename TConfig>
inline TConfig PassConfigWithDefaults() {
  static_assert(std::is_base_of_v<ffi::ObjectRef, TConfig>,
                "Can only create ObjectRef-derived types");
  using ContainerType = typename TConfig::ContainerType;
  static auto finit_object = ffi::Function::GetGlobalRequired("ffi.MakeObjectFromPackedArgs");
  ffi::AnyView packed_args[1];
  packed_args[0] = ContainerType::RuntimeTypeIndex();
  ffi::Any rv;
  finit_object.CallPacked(ffi::PackedArgs(packed_args, 1), &rv);
  return rv.cast<TConfig>();
}

/*!
 * \brief Meta data that will be used to help optimization and analysis.
 * \sa PassInfo
 */
class PassInfoNode : public ffi::Object {
 public:
  /*! \brief The minimal optimization level that this pass will be enabled. */
  int opt_level;

  /*! \brief The name of an optimization/analysis pass. */
  ffi::String name;

  PassInfoNode() = default;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PassInfoNode>()
        .def_ro("opt_level", &PassInfoNode::opt_level)
        .def_ro("name", &PassInfoNode::name);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("transform.PassInfo", PassInfoNode, ffi::Object);
};

/*!
 * \brief Managed reference class for PassInfoNode
 * \sa PassInfoNode
 */
class PassInfo : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param opt_level The optimization level
   * \param name Name of the pass.
   */
  TVM_DLL PassInfo(int opt_level, ffi::String name);

  explicit PassInfo(ffi::ObjectPtr<PassInfoNode> n) : ffi::ObjectRef(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(n != nullptr);
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PassInfo, ffi::ObjectRef, PassInfoNode);
};

/*!
 * \brief PassNode is the base type of differnt types of optimization passes.
 * It is designed as a pure class and implemented by different pass subclasses
 * at different granularity of Relax nodes.
 */
class PassNode : public ffi::Object {
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
  virtual IRModule operator()(IRModule mod, const PassContext& pass_ctx) const = 0;
  TVM_FFI_DECLARE_OBJECT_INFO("transform.Pass", PassNode, ffi::Object);
};

class Pass : public ffi::ObjectRef {
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
  IRModule operator()(IRModule mod) const;

  /*!
   * \brief Transform mod using a functor under a given pass context.
   *
   * \param mod The module that an optimization pass runs on.
   * \param pass_ctx The pass context that can provide information for the optimization.
   *
   * \return The transformed module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const;

  explicit Pass(ffi::ObjectPtr<PassNode> n) : ffi::ObjectRef(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(n != nullptr);
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Pass, ffi::ObjectRef, PassNode);

 private:
  IRModule static AssertImmutableModule(const IRModule& mod, const PassNode* node,
                                        const PassContext& pass_ctx);
};

/*!
 * \brief The SequentialNode contains a set of passes that transform Relax
 * programs from one AST to another semantically equivalent one.
 *
 * One example of this level of pass is that the pass manager needs to correctly
 * perform a host of optimizations with a given optimization level and disabled
 * passes.
 */
class SequentialNode : public PassNode {
 public:
  explicit SequentialNode(PassInfo pass_info) : pass_info(std::move(pass_info)) {}
  explicit SequentialNode(ffi::UnsafeInit tag) : pass_info(tag) {}
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief A list of passes that used to compose a sequential pass. */
  tvm::ffi::Array<Pass> passes;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SequentialNode>()
        .def_ro("pass_info", &SequentialNode::pass_info)
        .def_ro("passes", &SequentialNode::passes);
  }

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }

  /*!
   * \brief Perform optimizations on a series of passes. The aforementioned
   *        typical pass manager jobs could be done by it. This function could
   *        be overloaded to focus on different metrics, i.e. performance,
   *        memory footprint, etc.
   *
   * \param mod The module that these passes are applied on.
   * \param pass_ctx The context that these passes execute on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("transform.Sequential", SequentialNode, PassNode);
};

class Sequential : public Pass {
 public:
  /*!
   * \brief The constructor of `Sequential`.
   *
   * \param passes The passes to apply.
   * \param pass_info The pass metadata.
   */
  TVM_DLL Sequential(ffi::Array<Pass> passes, PassInfo pass_info);

  /*!
   * \brief The constructor of `Sequential`.
   *
   * \param passes The passes to apply.
   * \param name The name of a sequential pass. It's defaulted to "sequential".
   *        This allows users to only provide a list of passes and execute them
   *        under a given context.
   */
  TVM_DLL Sequential(ffi::Array<Pass> passes, ffi::String name = "sequential");

  explicit Sequential(ffi::UnsafeInit tag) : Pass(tag) {}
  explicit Sequential(ffi::ObjectPtr<SequentialNode> n) : Pass(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(n != nullptr);
    data_ = std::move(n);
  }

  const SequentialNode* operator->() const;
  using ContainerType = SequentialNode;
};

/*
 * \brief Create a module pass.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the module pass.
 * \param name The name of the module pass.
 *
 * \return The created module pass.
 */
TVM_DLL Pass CreateModulePass(std::function<IRModule(IRModule, PassContext)> pass_func,
                              int opt_level, ffi::String name);

/*!
 * \brief A special trace pass that prints the header and IR to LOG(INFO).
 * \param header The header to be attached to the output.
 * \return The pass.
 */
TVM_DLL Pass PrintIR(ffi::String header = "");

/*!
 * \brief Enrich a pass-time error with a TVMScript-rendered, underlined source
 *        location derived from the error's embedded VisitErrorContext.
 *
 * Returns an ffi::Error that preserves err's kind, original message, and
 * backtrace, and appends the failing pass name plus the offending location
 * rendered as TVMScript (the whole \p mod, or local to \p func when provided).
 * The returned error drops the VisitErrorContext payload, so an outer catch
 * that re-enriches finds no context and returns the error unchanged.
 *
 * Pure and total: never throws; returns \p err unchanged when there is no
 * context, the path is unresolvable, or rendering fails.
 *
 * \param err The error thrown by the pass body.
 * \param mod The IRModule the pass ran on (the access-path root, or the
 *            container of \p func when \p func is provided).
 * \param pass_name The name of the failing pass, shown in the message.
 * \param func When set, resolve and render the location local to
 *             \p mod->functions[func]; otherwise use the whole module.
 * \return The enriched (or, on any fallback, the original) error.
 */
TVM_DLL ffi::Error EnrichPassErrorWithContext(
    const ffi::Error& err, const IRModule& mod, ffi::String pass_name,
    ffi::Optional<GlobalVar> func = ffi::Optional<GlobalVar>(std::nullopt));

/*!
 * \brief PassInstrumentNode forms an instrument implementation.
 * It provides API for users to register callbacks at different instrumentation points.
 *
 * Within a PassContext, call sequence of a PassInstrument implementation is like:
 *
 *   with PassContext(instruments=[pi]):  # pi = a PassInstrument implementation
 *       pi.EnterPassContext()
 *
 *       if pi.ShouldRun(Pass1):
 *           pi.RunBeforePass()
 *           Pass1()
 *           pi.RunAfterPass()
 *
 *       if pi.ShouldRun(Pass2):
 *           pi.RunBeforePass()
 *           Pass2()
 *           pi.RunAfterPass()
 *
 *       pi.ExitPassContext()
 *
 * `EnterPassContext` and `ExitPassContext` are only called once when entering/exiting a
 * PassContext. `ShouldRun`, `RunBeforePass` and `RunAfterPass` are called multiple times depending
 * on how many passes.
 *
 * If there are multiple pass instrumentations provided, the instrument points are the same.
 * PassInstrument implementations' callbacks are called in order:
 *
 *   with PassContext(instruments=[pi1, pi2]):  # pi1, pi2 = two distinct PassInstrument impls
 *       pi.EnterPassContext() for pi in instruments
 *
 *       should_run = all([pi.ShoudRun(Pass1) for pi in instruments)])
 *       if (should_run)
 *           pi.RunBeforePass() for pi in instruments
 *           Pass1()
 *           pi.RunAfterPass()  for pi in instruments
 *
 *       should_run = all([pi.ShouldRun(Pass2) for pi in instruments)])
 *       if (should_run)
 *           pi.RunBeforePass() for pi in instruments
 *           Pass2()
 *           pi.RunAfterPass() for pi in instruments
 *
 *       pi.ExitPassContext() for pi in instruments
 *
 * Note:
 *   1. Assume there is no dependency between PassInstrument implementations in `instruments` .
 *   2. `EnterPassContext` and `ExitPassContext` have `with` behavior (see PassContext and its FFI):
 *        If there is any exception raised in `ShouldRun()`, `RunBeforePass()`, `RunAfterPass()` and
 *        `Pass()`, `ExitPassContext()` is still called.
 *   3. In mutiple PassInstrument instances scenario, callbacks are called in order:
 *        If one throws exceptions, remainings will not be called.
 *
 * \sa PassInstrument
 * \sa src/ir/transform.cc
 */
class PassInstrumentNode : public ffi::Object {
 public:
  /*! \brief Name of this pass instrument object. */
  ffi::String name;

  virtual ~PassInstrumentNode() {}

  /*! \brief Instrument when entering PassContext. Called once within a PassContext. */
  virtual void EnterPassContext() const = 0;

  /*! \brief Instrument when exiting PassContext. Called once within a PassContext. */
  virtual void ExitPassContext() const = 0;

  /*!
   * \brief Determine whether to run the pass or not. Called multiple times depend on number of
   *        passes.
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   *
   * \return true to run the pass; false to skip the pass.
   */
  virtual bool ShouldRun(const IRModule& mod, const transform::PassInfo& info) const = 0;

  /*!
   * \brief Instrument before pass run. Called multiple times depend on number of passes.
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   */
  virtual void RunBeforePass(const IRModule& mod, const transform::PassInfo& info) const = 0;

  /*!
   * \brief Instrument after pass run. Called multiple time depend on number of passes.
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   */
  virtual void RunAfterPass(const IRModule& mod, const transform::PassInfo& info) const = 0;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PassInstrumentNode>().def_ro("name", &PassInstrumentNode::name);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("transform.PassInstrument", PassInstrumentNode, ffi::Object);
};

/*!
 * \brief Managed reference class for PassInstrumentNode
 * \sa PassInstrumentNode
 */
class PassInstrument : public ffi::ObjectRef {
 public:
  explicit PassInstrument(ffi::ObjectPtr<PassInstrumentNode> n)
      : ffi::ObjectRef(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(n != nullptr);
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PassInstrument, ffi::ObjectRef, PassInstrumentNode);
};

/*!
 * \brief PassContextNode contains the information that a pass can rely on,
 * such as analysis results.
 * \sa PassContext
 */
class PassContextNode : public ffi::Object {
 public:
  /*! \brief The default optimization level. */
  int opt_level{2};

  /*! \brief The list of required passes. */
  ffi::Array<ffi::String> required_pass;
  /*! \brief The list of disabled passes. */
  ffi::Array<ffi::String> disabled_pass;
  /*! \brief Pass specific configurations. */
  ffi::Map<ffi::String, Any> config;

  /*! \brief A list of pass instrument implementations. */
  ffi::Array<PassInstrument> instruments;

  PassContextNode() = default;

  /*!
   * \brief Get a config value, returning an empty optional when the key is absent.
   * \param key The config key.
   * \tparam T The expected value type.
   * \throw Error if the key exists but the value does not match T.
   */
  template <typename T>
  ffi::Optional<T> GetConfig(const std::string& key) const {
    if (config.defined()) {
      auto it = config.find(key);
      if (it != config.end()) {
        return (*it).second.as_or_throw<ffi::Optional<T>>();
      }
    }
    return std::nullopt;
  }

  /*!
   * \brief Get a config object, constructing reflection defaults only when absent.
   * \param key The config key.
   * \tparam TConfig The ObjectRef config type with reflection-defined defaults.
   * \return The configured object, or a fresh default object for this call.
   */
  template <typename TConfig>
  TConfig GetConfigOrDefault(const std::string& key) const {
    auto config = GetConfig<TConfig>(key);
    if (config.has_value()) {
      return config.value();
    }
    return PassConfigWithDefaults<TConfig>();
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PassContextNode>()
        .def_ro("opt_level", &PassContextNode::opt_level)
        .def_ro("required_pass", &PassContextNode::required_pass)
        .def_ro("disabled_pass", &PassContextNode::disabled_pass)
        .def_ro("instruments", &PassContextNode::instruments)
        .def_ro("config", &PassContextNode::config);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("transform.PassContext", PassContextNode, ffi::Object);
};

inline PassContext::PassContext(ffi::ObjectPtr<PassContextNode> n) : ffi::ObjectRef(n) {
  TVM_FFI_ICHECK(n != nullptr);
}

inline const PassContextNode* PassContext::operator->() const {
  TVM_FFI_ICHECK(get() != nullptr);
  return static_cast<const PassContextNode*>(get());
}

inline PassContextNode* PassContext::operator->() {
  TVM_FFI_ICHECK(get() != nullptr);
  return static_cast<PassContextNode*>(get_mutable());
}

}  // namespace transform
}  // namespace tvm

#endif  // TVM_IR_TRANSFORM_H_
