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
#ifndef TVM_META_SCHEDULE_RUNNER_H_
#define TVM_META_SCHEDULE_RUNNER_H_

#include <tvm/ir/expr.h>

namespace tvm {
namespace meta_schedule {

/*! \brief The runner's result. */
class RunnerResultNode : public runtime::Object {
 public:
  /*! \brief The run time in seconds.*/
  Optional<Array<FloatImm>> run_secs;
  /*! \brief The error message, if any. */
  Optional<String> error_msg;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("run_secs", &run_secs);
    v->Visit("error_msg", &error_msg);
  }

  static constexpr const char* _type_key = "meta_schedule.RunnerResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(RunnerResultNode, runtime::Object);
};

/*!
 * \brief Managed reference to RunnerResultNode
 * \sa RunnerResultNode
 */
class RunnerResult : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor for RunnerResult.
   * \param run_secs The run time in seconds.
   * \param error_msg The error message, if any.
   */
  TVM_DLL explicit RunnerResult(Optional<Array<FloatImm>> run_secs, Optional<String> error_msg);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(RunnerResult, runtime::ObjectRef, RunnerResultNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_RUNNER_H_
