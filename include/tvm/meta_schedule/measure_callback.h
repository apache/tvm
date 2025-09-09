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

#ifndef TVM_META_SCHEDULE_MEASURE_CALLBACK_H_
#define TVM_META_SCHEDULE_MEASURE_CALLBACK_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/meta_schedule/builder.h>
#include <tvm/meta_schedule/measure_candidate.h>
#include <tvm/meta_schedule/runner.h>
#include <tvm/meta_schedule/search_strategy.h>
#include <tvm/meta_schedule/tune_context.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace meta_schedule {

class TaskScheduler;

/*! \brief Rules to apply after measure results is available. */
class MeasureCallbackNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~MeasureCallbackNode() = default;

  static void RegisterReflection() {
    // No fields to register
  }

  /*!
   * \brief Apply a measure callback rule with given arguments.
   * \param task_scheduler The task scheduler.
   * \param task_id The id of the task (tune context) to apply measure callbacks.
   * \param measure_candidates The measure candidates.
   * \param builder_results The builder results by building the measure candidates.
   * \param runner_results The runner results by running the built measure candidates.
   */
  virtual void Apply(const TaskScheduler& task_scheduler,                     //
                     int task_id,                                             //
                     const ffi::Array<MeasureCandidate>& measure_candidates,  //
                     const ffi::Array<BuilderResult>& builder_results,        //
                     const ffi::Array<RunnerResult>& runner_results) = 0;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("meta_schedule.MeasureCallback", MeasureCallbackNode, Object);
};

/*! \brief The measure callback with customized methods on the python-side. */
class PyMeasureCallbackNode : public MeasureCallbackNode {
 public:
  /*!
   * \brief Apply a measure callback to the given schedule.
   * \param task_scheduler The task scheduler.
   * \param tasks The list of tune context to process.
   * \param measure_candidates The measure candidates.
   * \param builds The builder results by building the measure candidates.
   * \param results The runner results by running the built measure candidates.
   * \return Whether the measure callback was successfully applied.
   */
  using FApply = ffi::TypedFunction<void(const TaskScheduler& task_scheduler,                     //
                                         int task_id,                                             //
                                         const ffi::Array<MeasureCandidate>& measure_candidates,  //
                                         const ffi::Array<BuilderResult>& builds,                 //
                                         const ffi::Array<RunnerResult>& results)>;
  /*!
   * \brief Get the measure callback function as string with name.
   * \return The string of the measure callback function.
   */
  using FAsString = ffi::TypedFunction<ffi::String()>;

  /*! \brief The packed function to the `Apply` function. */
  FApply f_apply;
  /*! \brief The packed function to the `AsString` function. */
  FAsString f_as_string;

  static void RegisterReflection() {
    // `f_apply` is not registered
    // `f_as_string` is not registered
  }

  void Apply(const TaskScheduler& task_scheduler,                     //
             int task_id,                                             //
             const ffi::Array<MeasureCandidate>& measure_candidates,  //
             const ffi::Array<BuilderResult>& builds,                 //
             const ffi::Array<RunnerResult>& results);
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("meta_schedule.PyMeasureCallback", PyMeasureCallbackNode,
                                    MeasureCallbackNode);
};

/*!
 * \brief Managed reference to MeasureCallbackNode
 * \sa MeasureCallbackNode
 */
class MeasureCallback : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a measure callback that adds the measurement results into the database
   * \return The measure callback created.
   */
  TVM_DLL static MeasureCallback AddToDatabase();
  /*!
   * \brief Create a measure callback that removes the build artifacts from the disk
   * \return The measure callback created.
   */
  TVM_DLL static MeasureCallback RemoveBuildArtifact();
  /*!
   * \brief Create a measure callback that updates the cost model with measurement result.
   * \return The measure callback created.
   */
  TVM_DLL static MeasureCallback UpdateCostModel();
  /*!
   * \brief Create a measure callback with customized methods on the python-side.
   * \param f_apply The packed function of `Apply`.
   * \param f_as_string The packed function of `AsString`.
   * \return The measure callback created.
   */
  TVM_DLL static MeasureCallback PyMeasureCallback(PyMeasureCallbackNode::FApply f_apply,
                                                   PyMeasureCallbackNode::FAsString f_as_string);
  /*! \brief The default list of measure callbacks. */
  TVM_DLL static ffi::Array<MeasureCallback, void> Default();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MeasureCallback, ObjectRef, MeasureCallbackNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_MEASURE_CALLBACK_H_
