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
 * \brief The user interface of the auto-scheduler
 */

#ifndef TVM_ANSOR_AUTO_SCHEDULE_H_
#define TVM_ANSOR_AUTO_SCHEDULE_H_

#include <utility>
#include <string>
#include "measure.h"
#include "search_policy/search_policy.h"

namespace tvm {
namespace ansor {

/*! \brief Tuning and measurement options */
class TuneOptionNode : public Object {
 public:
  int n_trials;              // Number of total measurement trials
  int early_stopping;        // Stops early the tuning if no improvement after n measurements
  int num_measure_per_iter;  // The number of programs to be measured at each iteration
  int verbose;               // Verbosity level. 0 means silent.
  Builder builder;           // Builder which builds the program
  Runner runner;             // Runner which runs the program and measure time costs
  Array<MeasureCallback> measure_callbacks;    // MeasureCallback functions
  Array<SearchCallback> pre_search_callbacks;  // SearchCallback functions
                                               // run before search

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("n_trials", &n_trials);
    v->Visit("early_stopping", &early_stopping);
    v->Visit("num_measure_per_iter", &num_measure_per_iter);
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
  TuneOption(int n_trials, int early_stopping, int num_measure_per_iter,
             int verbose, Builder builder, Runner runner,
             Array<MeasureCallback> measure_callbacks,
             Array<SearchCallback> pre_search_callbacks);

  TVM_DEFINE_OBJECT_REF_METHODS(TuneOption, ObjectRef, TuneOptionNode);
};

/*! \brief Auto schedule for a compute declaration */
std::pair<te::Schedule, Array<te::Tensor> > AutoSchedule(
    SearchTask task, SearchPolicy search_policy, TuneOption tune_option);

/*! \brief Auto schedule for a compute declaration */
std::pair<te::Schedule, Array<te::Tensor> > AutoSchedule(
    std::string workload_key, Target target, Target target_host,
    SearchPolicy search_policy, HardwareParams hardware_params,
    TuneOption tune_option);

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_AUTO_SCHEDULE_H_
