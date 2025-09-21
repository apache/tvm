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

#ifndef TVM_META_SCHEDULE_MEASURE_CANDIDATE_H_
#define TVM_META_SCHEDULE_MEASURE_CANDIDATE_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/meta_schedule/arg_info.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace meta_schedule {

/*! \brief The schedule (with input shapes) to be measured. */
class MeasureCandidateNode : public runtime::Object {
 public:
  /*! \brief The schedule for measurement. */
  tir::Schedule sch;
  /*! \brief The argument information, e.g., (shape, dtype) for tensors. */
  ffi::Array<ArgInfo> args_info;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MeasureCandidateNode>()
        .def_ro("sch", &MeasureCandidateNode::sch)
        .def_ro("args_info", &MeasureCandidateNode::args_info);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("meta_schedule.MeasureCandidate", MeasureCandidateNode, Object);
};

/*!
 * \brief Managed reference to MeasureCandidateNode.
 * \sa MeasureCandidateNode
 */
class MeasureCandidate : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor of MeasureCandidate.
   * \param sch The schedule for measurement.
   * \param args_info The argument information, e.g., (shape, dtype) for tensors.
   */
  TVM_DLL MeasureCandidate(tir::Schedule sch, ffi::Array<ArgInfo> args_info);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MeasureCandidate, ObjectRef, MeasureCandidateNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_MEASURE_CANDIDATE_H_
