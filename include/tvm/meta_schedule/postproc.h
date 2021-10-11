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

#ifndef TVM_META_SCHEDULE_POSTPROC_H_
#define TVM_META_SCHEDULE_POSTPROC_H_

#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace meta_schedule {

class TuneContext;

/*!
 * \brief Rules to apply a post processing to a schedule.
 * \note Post processing is designed to deal with the problem of undertermined schedule validity
 *  after applying some schedule primitve at runtime. E.g., Fuse the first X loops to reach the
 *  maximum number below 1024, X is only decided at runtime.
 */
class PostprocNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~PostprocNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param tune_context The tuning context for initialization.
   */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Apply a post processing to the given schedule.
   * \param sch The schedule to be post processed.
   * \return Whether the post processing was successfully applied.
   */
  virtual bool Apply(const tir::Schedule& schedule) = 0;

  static constexpr const char* _type_key = "meta_schedule.Postproc";
  TVM_DECLARE_BASE_OBJECT_INFO(PostprocNode, Object);
};

/*! \brief The post processing with customized methods on the python-side. */
class PyPostprocNode : public PostprocNode {
 public:
  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param tune_context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  /*!
   * \brief Apply a post processing to the given schedule.
   * \param sch The schedule to be post processed.
   * \return Whether the post processing was successfully applied.
   */
  using FApply = runtime::TypedPackedFunc<bool(const tir::Schedule&)>;

  /*! \brief The packed function to the `InitializeWithTuneContext` funcion. */
  FInitializeWithTuneContext f_initialize_with_tune_context;
  /*! \brief The packed function to the `Apply` funcion. */
  FApply f_apply;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_initialize_with_tune_context` is not visited
    // `f_apply` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    this->f_initialize_with_tune_context(context);
  }

  bool Apply(const tir::Schedule& sch) final { return this->f_apply(sch); }

  static constexpr const char* _type_key = "meta_schedule.PyPostproc";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyPostprocNode, PostprocNode);
};

/*!
 * \brief Managed reference to PostprocNode
 * \sa PostprocNode
 */
class Postproc : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a post processing with customized methods on the python-side.
   * \param f_initialize_with_tune_context The packed function of `InitializeWithTuneContext`.
   * \param f_apply The packed function of `Apply`.
   * \return The post processing created.
   */
  TVM_DLL static Postproc PyPostproc(
      PyPostprocNode::FInitializeWithTuneContext f_initialize_with_tune_context,
      PyPostprocNode::FApply f_apply);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Postproc, ObjectRef, PostprocNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_POSTPROC_H_
