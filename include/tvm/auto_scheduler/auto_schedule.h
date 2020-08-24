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
 * \file tvm/auto_scheduler/auto_schedule.h
 * \brief The user interface of the auto scheduler.
 */

#ifndef TVM_AUTO_SCHEDULER_AUTO_SCHEDULE_H_
#define TVM_AUTO_SCHEDULER_AUTO_SCHEDULE_H_

#include <tvm/auto_scheduler/measure.h>
#include <tvm/auto_scheduler/search_policy.h>

#include <utility>

namespace tvm {
namespace auto_scheduler {

/*! \brief Tuning and measurement options. */
class TuningOptionsNode : public Object {
 public:
  /*! \brief The number of total measurement trials. */
  int num_measure_trials;
  /*! \brief Stops the tuning early if no improvement after n measurements. */
  int early_stopping;
  /*! \brief The number of programs to be measured at each search round. */
  int num_measures_per_round;
  /*! \brief Verbosity level. 0 for silent, 1 to output information during schedule searching. */
  int verbose;
  /*! \brief ProgramBuilder which builds the program */
  ProgramBuilder builder;
  /*! \brief ProgramRunner which runs the program and measures time costs */
  ProgramRunner runner;
  /*! \brief MeasureCallback functions to be called after each measure batch */
  Optional<Array<MeasureCallback>> measure_callbacks;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_measure_trials", &num_measure_trials);
    v->Visit("early_stopping", &early_stopping);
    v->Visit("num_measures_per_round", &num_measures_per_round);
    v->Visit("verbose", &verbose);
    v->Visit("builder", &builder);
    v->Visit("runner", &runner);
    v->Visit("measure_callbacks", &measure_callbacks);
  }

  static constexpr const char* _type_key = "auto_scheduler.TuningOptions";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuningOptionsNode, Object);
};

/*!
 * \brief Managed reference to TuningOptionsNode.
 * \sa TuningOptionsNode
 */
class TuningOptions : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param num_measure_trials The number of total measurement trials.
   * \param early_stopping Stops the tuning early if no improvement after n measurements.
   * \param num_measures_per_round The number of programs to be measured at each search round.
   * \param verbose Verbosity level. 0 for silent, 1 to output information during schedule
   * search.
   * \param builder ProgramBuilder which builds the program.
   * \param runner ProgramRunner which runs the program and measure time costs.
   * \param measure_callbacks MeasureCallback functions to be called after each measure batch.
   */
  TuningOptions(int num_measure_trials, int early_stopping, int num_measures_per_round, int verbose,
                ProgramBuilder builder, ProgramRunner runner,
                Optional<Array<MeasureCallback>> measure_callbacks);

  TVM_DEFINE_OBJECT_REF_METHODS(TuningOptions, ObjectRef, TuningOptionsNode);
};

/*!
 * \brief Run schedule search for a given compute declaration.
 * \param search_policy The search policy.
 * \param tuning_options Tuning and measurement options.
 * \return A `te::schedule` and an Array of `te::Tensor` to be used in `tvm.lower` or
 * `tvm.build`.
 */
TVM_DLL std::pair<te::Schedule, Array<te::Tensor>> AutoSchedule(SearchPolicy search_policy,
                                                                TuningOptions tuning_options);
}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_AUTO_SCHEDULE_H_
