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
#ifndef TVM_META_SCHEDULE_TUNE_CONTEXT_H_
#define TVM_META_SCHEDULE_TUNE_CONTEXT_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/meta_schedule/builder.h>
#include <tvm/meta_schedule/runner.h>
#include <tvm/meta_schedule/search_strategy.h>
#include <tvm/meta_schedule/space_generator.h>
#include <tvm/runtime/object.h>
#include <tvm/support/random_engine.h>
#include <tvm/target/target.h>

namespace tvm {
namespace meta_schedule {

class TaskSchedulerNode;
class MeasureCallback;
class TuneContext;

/*! \brief The auto tuning context. */
class TuneContextNode : public runtime::Object {
 public:
  using TRandState = support::LinearCongruentialEngine::TRandState;

  /*! \brief The workload to be tuned. */
  ffi::Optional<IRModule> mod;
  /*! \brief The target to be tuned for. */
  ffi::Optional<Target> target;
  /*! \brief The design space generator. */
  ffi::Optional<SpaceGenerator> space_generator;
  /*! \brief The search strategy. */
  ffi::Optional<SearchStrategy> search_strategy;
  /*! \brief The name of the tuning task. */
  ffi::Optional<ffi::String> task_name;
  /*! \brief The number of threads to be used. */
  int num_threads;
  /*! \brief The random state. */
  TRandState rand_state;
  /*! \brief The tuning task's logging function. t*/
  ffi::Function logger;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TuneContextNode>()
        .def_ro("mod", &TuneContextNode::mod)
        .def_ro("target", &TuneContextNode::target)
        .def_ro("space_generator", &TuneContextNode::space_generator)
        .def_ro("search_strategy", &TuneContextNode::search_strategy)
        .def_ro("task_name", &TuneContextNode::task_name)
        .def_ro("num_threads", &TuneContextNode::num_threads)
        .def_ro("rand_state", &TuneContextNode::rand_state);
    // `logger` is not registered
  }

  /*!
   * \brief Initialize members that needs initialization with tune context.
   */
  void Initialize();
  /*!
   * \brief Clone the tune context.
   * \return The cloned tune context.
   */
  TuneContext Clone() const;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("meta_schedule.TuneContext", TuneContextNode, Object);
};

/*!
 * \brief Managed reference to TuneContextNode.
 * \sa TuneContextNode
 */
class TuneContext : public runtime::ObjectRef {
 public:
  using TRandState = support::LinearCongruentialEngine::TRandState;
  /*!
   * \brief Constructor from ObjectPtr<TuneContextNode>.
   * \param data The object pointer.
   */
  explicit TuneContext(ObjectPtr<TuneContextNode> data) : ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  /*!
   * \brief Constructor.
   * \param mod The workload to be tuned.
   * \param target The target to be tuned for.
   * \param space_generator The design space generator.
   * \param search_strategy The search strategy.
   * \param task_name The name of the tuning task.
   * \param num_threads The number of threads to be used.
   * \param rand_state The random state.
   * \param logger The tuning task's logging function.
   */
  TVM_DLL explicit TuneContext(ffi::Optional<IRModule> mod, ffi::Optional<Target> target,
                               ffi::Optional<SpaceGenerator> space_generator,
                               ffi::Optional<SearchStrategy> search_strategy,
                               ffi::Optional<ffi::String> task_name, int num_threads,
                               TRandState rand_state, ffi::Function logger);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TuneContext, ObjectRef, TuneContextNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_TUNE_CONTEXT_H_
