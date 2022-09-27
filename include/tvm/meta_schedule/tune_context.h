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
#include <tvm/meta_schedule/mutator.h>
#include <tvm/meta_schedule/postproc.h>
#include <tvm/meta_schedule/runner.h>
#include <tvm/meta_schedule/schedule_rule.h>
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
  /*! \brief The workload to be tuned. */
  Optional<IRModule> mod;
  /*! \brief The target to be tuned for. */
  Optional<Target> target;
  /*! \brief The design space generator. */
  Optional<SpaceGenerator> space_generator;
  /*! \brief The search strategy. */
  Optional<SearchStrategy> search_strategy;
  /*! \brief The schedule rules. */
  Array<ScheduleRule> sch_rules;
  /*! \brief The postprocessors. */
  Array<Postproc> postprocs;
  /*! \brief The probability of using certain mutator. */
  Map<Mutator, FloatImm> mutator_probs;
  /*! \brief The name of the tuning task. */
  Optional<String> task_name;
  /*! \brief The tuning task's logging function. t*/
  PackedFunc logging_func;
  /*! \brief The random state. */
  support::LinearCongruentialEngine::TRandState rand_state;
  /*! \brief The number of threads to be used. */
  int num_threads;

  /*! \brief Whether the tuning task has been stopped or finished. */
  bool is_terminated;  // TODO(@junrushao1994): move to TaskScheduler
  /*! \brief The measure candidates. */
  Optional<Array<MeasureCandidate>> measure_candidates;
  /*! \brief The building results. */
  Optional<Array<BuilderResult>> builder_results;
  /*! \brief Packed functions to fetch the runner results asynchronously. */
  Optional<Array<RunnerFuture>> runner_futures;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("mod", &mod);
    v->Visit("target", &target);
    v->Visit("space_generator", &space_generator);
    v->Visit("search_strategy", &search_strategy);
    v->Visit("sch_rules", &sch_rules);
    v->Visit("postprocs", &postprocs);
    v->Visit("mutator_probs", &mutator_probs);
    v->Visit("task_name", &task_name);
    // `logging_func` is not visited
    v->Visit("rand_state", &rand_state);
    v->Visit("num_threads", &num_threads);
    v->Visit("is_terminated", &is_terminated);
    v->Visit("measure_candidates", &measure_candidates);
    v->Visit("builder_results", &builder_results);
    v->Visit("runner_futures", &runner_futures);
  }

  /*! \brief Initialize members that needs initialization with tune context. */
  void Initialize();
  /*!
   * \brief Clone the tune context.
   * \return The cloned tune context.
   */
  TuneContext Clone() const;
  /*! \brief Set the measure candidates from the SearchStrategy */
  void _SetMeasureCandidates(const Array<MeasureCandidate>& candidates);
  /*!
   * \brief Send the measure candidates to builder.
   * \param builder The builder to send the candidates to.
   */
  void _SendToBuilder(const Builder& builder);
  /*!
   * \brief Send the built measure candidates to runner.
   * \param runner The runner to send the candidates to.
   */
  void _SendToRunner(const Runner& runner);
  /*!
   * \brief Join the running tasks.
   * \returns The results from the runner
   */
  Array<RunnerResult> _Join();
  /*! \brief Set `measure_candidates`, `builder_results` and `runner_futures` to null. */
  void _ClearMeasureState();
  static constexpr const char* _type_key = "meta_schedule.TuneContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuneContextNode, Object);
};

/*!
 * \brief Managed reference to TuneContextNode.
 * \sa TuneContextNode
 */
class TuneContext : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor.
   * \param mod The workload to be tuned.
   * \param target The target to be tuned for.
   * \param space_generator The design space generator.
   * \param search_strategy The search strategy.
   * \param sch_rules The schedule rules.
   * \param postprocs The postprocessors.
   * \param mutator_probs The probability of using certain mutator.
   * \param task_name The name of the tuning task.
   * \param logging_func The tuning task's logging function.
   * \param rand_state The random state.
   * \param num_threads The number of threads to be used.
   */
  TVM_DLL explicit TuneContext(Optional<IRModule> mod,                                    //
                               Optional<Target> target,                                   //
                               Optional<SpaceGenerator> space_generator,                  //
                               Optional<SearchStrategy> search_strategy,                  //
                               Optional<Array<ScheduleRule>> sch_rules,                   //
                               Optional<Array<Postproc>> postprocs,                       //
                               Optional<Map<Mutator, FloatImm>> mutator_probs,            //
                               Optional<String> task_name,                                //
                               PackedFunc logging_func,                                   //
                               support::LinearCongruentialEngine::TRandState rand_state,  //
                               int num_threads);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TuneContext, ObjectRef, TuneContextNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_TUNE_CONTEXT_H_
