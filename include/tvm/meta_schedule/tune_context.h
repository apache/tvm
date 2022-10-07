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

#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/meta_schedule/builder.h>
#include <tvm/meta_schedule/runner.h>
#include <tvm/meta_schedule/search_strategy.h>
#include <tvm/meta_schedule/space_generator.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
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
  Optional<IRModule> mod;
  /*! \brief The target to be tuned for. */
  Optional<Target> target;
  /*! \brief The design space generator. */
  Optional<SpaceGenerator> space_generator;
  /*! \brief The search strategy. */
  Optional<SearchStrategy> search_strategy;
  /*! \brief The name of the tuning task. */
  Optional<String> task_name;
  /*! \brief The number of threads to be used. */
  int num_threads;
  /*! \brief The random state. */
  TRandState rand_state;
  /*! \brief The tuning task's logging function. t*/
  PackedFunc logger;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("mod", &mod);
    v->Visit("target", &target);
    v->Visit("space_generator", &space_generator);
    v->Visit("search_strategy", &search_strategy);
    v->Visit("task_name", &task_name);
    v->Visit("num_threads", &num_threads);
    v->Visit("rand_state", &rand_state);
    // `logger` is not visited
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

  static constexpr const char* _type_key = "meta_schedule.TuneContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuneContextNode, Object);
};

/*!
 * \brief Managed reference to TuneContextNode.
 * \sa TuneContextNode
 */
class TuneContext : public runtime::ObjectRef {
 public:
  using TRandState = support::LinearCongruentialEngine::TRandState;
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
  TVM_DLL explicit TuneContext(Optional<IRModule> mod, Optional<Target> target,
                               Optional<SpaceGenerator> space_generator,
                               Optional<SearchStrategy> search_strategy, Optional<String> task_name,
                               int num_threads, TRandState rand_state, PackedFunc logger);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TuneContext, ObjectRef, TuneContextNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_TUNE_CONTEXT_H_
