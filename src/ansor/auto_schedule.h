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
 * \file ansor/auto_schedule.h
 * \brief The user interface of the Ansor auto-scheduler. This is the entry structure to get
 * schedule search requirements from upper level (Python API), and returns a high performance
 * schedule after search process.
 */

#ifndef TVM_ANSOR_AUTO_SCHEDULE_H_
#define TVM_ANSOR_AUTO_SCHEDULE_H_

#include <string>
#include <utility>

#include "measure.h"
#include "search_policy/search_policy.h"

namespace tvm {
namespace ansor {

/*! \brief Tuning and measurement options. */
class TuneOptionNode : public Object {
 public:
  /*! \brief Number of total measurement trials. */
  int n_trials;
  /*! \brief Stops early the tuning if no improvement after n measurements. */
  int early_stopping;
  /*! \brief The number of programs to be measured at each search round. */
  int num_measure_per_round;
  /*! \brief Verbosity level. (0 means silent) */
  int verbose;
  /*! \brief Builder which builds the program */
  Builder builder;
  /*! \brief Runner which runs the program and measure time costs */
  Runner runner;
  /*! \brief MeasureCallback functions to be called after each measure batch */
  Array<MeasureCallback> measure_callbacks;
  /*! \brief SearchCallback functions to be called before schedule search */
  Array<SearchCallback> pre_search_callbacks;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("n_trials", &n_trials);
    v->Visit("early_stopping", &early_stopping);
    v->Visit("num_measure_per_round", &num_measure_per_round);
    v->Visit("verbose", &verbose);
    v->Visit("builder", &builder);
    v->Visit("runner", &runner);
    v->Visit("measure_callbacks", &measure_callbacks);
    v->Visit("pre_search_callbacks", &pre_search_callbacks);
  }

  static constexpr const char* _type_key = "ansor.TuneOption";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuneOptionNode, Object);
};

/*!
 * \brief Managed reference to TuneOptionNode.
 * \sa TuneOptionNode
 */
class TuneOption : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param n_trials Number of total measurement trials.
   * \param early_stopping Stops early the tuning if no improvement after n measurements.
   * \param num_measure_per_round The number of programs to be measured at each search round.
   * \param verbose Verbosity level. (0 means silent)
   * \param builder Builder which builds the program.
   * \param runner Runner which runs the program and measure time costs.
   * \param measure_callbacks MeasureCallback functions to be called after each measure batch.
   * \param pre_search_callbacks SearchCallback functions to be called before schedule search.
   */
  TuneOption(int n_trials, int early_stopping, int num_measure_per_round, int verbose,
             Builder builder, Runner runner, Array<MeasureCallback> measure_callbacks,
             Array<SearchCallback> pre_search_callbacks);

  TVM_DEFINE_OBJECT_REF_METHODS(TuneOption, ObjectRef, TuneOptionNode);
};

/*!
 * \brief Auto schedule search for a given compute declaration, by SearchTask.
 * \param task The target search task.
 * \param search_policy The search policy to be used for schedule search.
 * \param tune_option Tuning and measurement options.
 * \return A `te::Schedule` and the target `te::Tensor` to be used in `tvm.lower` or `tvm.build`.
 */
std::pair<te::Schedule, Array<te::Tensor> > AutoSchedule(
    SearchTask task, SearchPolicy search_policy, TuneOption tune_option);

/*!
 * \brief Auto schedule search for a given compute declaration, by workload key.
 * \param workload_key The target workload key.
 * \param target The target device of this schedule search.
 * \param target_host The target host device of this schedule search.
 * \param search_policy The search policy to be used for schedule search.
 * \param hardware_params The hardware parameters of this schedule search.
 * \param tune_option Tuning and measurement options.
 * \return A `te::Schedule` and the target `te::Tensor` to be used in `tvm.lower` or `tvm.build`.
 */
std::pair<te::Schedule, Array<te::Tensor> > AutoSchedule(
    std::string workload_key, Target target, Target target_host,
    SearchPolicy search_policy, HardwareParams hardware_params,
    TuneOption tune_option);

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_AUTO_SCHEDULE_H_
