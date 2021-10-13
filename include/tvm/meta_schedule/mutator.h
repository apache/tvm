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

#ifndef TVM_META_SCHEDULE_MUTATOR_H_
#define TVM_META_SCHEDULE_MUTATOR_H_

#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace meta_schedule {

class TuneContext;

/*! \brief Mutator is designed to mutate the trace to explore the design space. */
class MutatorNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~MutatorNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param tune_context The tuning context for initialization.
   */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Apply the mutator function to the given trace.
   * \param trace The given trace for mutation.
   * \return None if mutator failed, otherwise return the mutated trace.
   */
  virtual Optional<tir::Trace> Apply(const tir::Trace& trace) = 0;

  static constexpr const char* _type_key = "meta_schedule.Mutator";
  TVM_DECLARE_BASE_OBJECT_INFO(MutatorNode, Object);
};

/*! \brief The mutator with customized methods on the python-side. */
class PyMutatorNode : public MutatorNode {
 public:
  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param tune_context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  /*!
   * \brief Apply the mutator function to the given trace.
   * \param trace The given trace for mutation.
   * \return None if mutator failed, otherwise return the mutated trace.
   */
  using FApply = runtime::TypedPackedFunc<Optional<tir::Trace>(const tir::Trace&)>;
  /*!
   * \brief Get the mutator as string with name.
   * \return The string of the mutator.
   */
  using FAsString = runtime::TypedPackedFunc<String()>;

  /*! \brief The packed function to the `InitializeWithTuneContext` funcion. */
  FInitializeWithTuneContext f_initialize_with_tune_context;
  /*! \brief The packed function to the `Apply` funcion. */
  FApply f_apply;
  /*! \brief The packed function to the `AsString` funcion. */
  FAsString f_as_string;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_initialize_with_tune_context` is not visited
    // `f_apply` is not visited
    // `f_as_string` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    this->f_initialize_with_tune_context(context);
  }

  Optional<tir::Trace> Apply(const tir::Trace& trace) final { return this->f_apply(trace); }

  static constexpr const char* _type_key = "meta_schedule.PyMutator";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyMutatorNode, MutatorNode);
};

/*!
 * \brief Managed reference to MutatorNode
 * \sa MutatorNode
 */
class Mutator : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a mutator with customized methods on the python-side.
   * \param f_initialize_with_tune_context The packed function of `InitializeWithTuneContext`.
   * \param f_apply The packed function of `Apply`.
   * \return The mutator created.
   */
  TVM_DLL static Mutator PyMutator(
      PyMutatorNode::FInitializeWithTuneContext f_initialize_with_tune_context,  //
      PyMutatorNode::FApply f_apply,                                             //
      PyMutatorNode::FAsString f_as_string);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Mutator, ObjectRef, MutatorNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_MUTATOR_H_
