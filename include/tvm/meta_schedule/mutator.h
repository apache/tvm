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

#include <tvm/node/reflection.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/support/random_engine.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/trace.h>

namespace tvm {
namespace meta_schedule {

class TuneContext;
class Mutator;

/*! \brief Mutator is designed to mutate the trace to explore the design space. */
class MutatorNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~MutatorNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Initialize the design space generator with tuning context.
   * \param context The tuning context for initialization.
   * \note This method is supposed to be called only once before every other method.
   */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Apply the mutator function to the given trace.
   * \param trace The given trace for mutation.
   * \param rand_state The random state for mutation.
   * \return None if mutator failed, otherwise return the mutated trace.
   */
  virtual Optional<tir::Trace> Apply(const tir::Trace& trace,
                                     support::LinearCongruentialEngine::TRandState* rand_state) = 0;

  /*!
   * \brief Clone the mutator.
   * \return The cloned mutator.
   */
  virtual Mutator Clone() const = 0;

  static constexpr const char* _type_key = "meta_schedule.Mutator";
  TVM_DECLARE_BASE_OBJECT_INFO(MutatorNode, Object);
};

/*!
 * \brief Managed reference to MutatorNode
 * \sa MutatorNode
 */
class Mutator : public runtime::ObjectRef {
 public:
  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  /*!
   * \brief Apply the mutator function to the given trace.
   * \param trace The given trace for mutation.
   * \return None if mutator failed, otherwise return the mutated trace.
   */
  using FApply = runtime::TypedPackedFunc<Optional<tir::Trace>(
      const tir::Trace&, support::LinearCongruentialEngine::TRandState rand_state)>;
  /*!
   * \brief Clone the mutator.
   * \return The cloned mutator.
   */
  using FClone = runtime::TypedPackedFunc<Mutator()>;
  /*!
   * \brief Get the mutator as string with name.
   * \return The string of the mutator.
   */
  using FAsString = runtime::TypedPackedFunc<String()>;
  /*! \brief Create a Mutator that mutates the decision of instruction Sample-Perfect-Tile */
  TVM_DLL static Mutator MutateTileSize();
  /*!
   * \brief Create a Mutator that mutates the parallel extent
   * \param max_jobs_per_core The maximum number of parallel jobs per core.
   * \return The created mutator.
   */
  TVM_DLL static Mutator MutateParallel(int64_t max_jobs_per_core);
  /*!
   * \brief Create a Mutator that mutates auto unroll step
   * \return The mutator created
   */
  TVM_DLL static Mutator MutateUnroll();
  /*!
   * \brief Create a Mutator that mutates the outcome of SampleComputeLocation
   * \return The mutator created
   */
  TVM_DLL static Mutator MutateComputeLocation();
  /*!
   * \brief Create a Mutator that mutates auto thread binding.
   * \return The mutator created
   */
  TVM_DLL static Mutator MutateThreadBinding();
  /*!
   * \brief Create a mutator with customized methods on the python-side.
   * \param f_initialize_with_tune_context The packed function of `InitializeWithTuneContext`.
   * \param f_apply The packed function of `Apply`.
   * \param f_clone The packed function of `Clone`.
   * \param f_as_string The packed function of `AsString`.
   * \return The mutator created.
   */
  TVM_DLL static Mutator PyMutator(FInitializeWithTuneContext f_initialize_with_tune_context,
                                   FApply f_apply, FClone f_clone, FAsString f_as_string);
  /*! \brief Create default mutators for LLVM */
  TVM_DLL static Map<Mutator, FloatImm, void> DefaultLLVM();
  /*! \brief Create default mutators for x86 VNNI */
  TVM_DLL static Map<Mutator, FloatImm, void> DefaultVNNI();
  /*! \brief Create default mutators for CUDA */
  TVM_DLL static Map<Mutator, FloatImm, void> DefaultCUDA();
  /*! \brief Create default mutators for CUDA with TensorCore */
  TVM_DLL static Map<Mutator, FloatImm, void> DefaultCUDATensorCore();
  /*! \brief Create default mutators for Hexagon */
  TVM_DLL static Map<Mutator, FloatImm, void> DefaultHexagon();

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Mutator, ObjectRef, MutatorNode);
};

/*! \brief The mutator with customized methods on the python-side. */
class PyMutatorNode : public MutatorNode {
 public:
  using FInitializeWithTuneContext = Mutator::FInitializeWithTuneContext;
  using FApply = Mutator::FApply;
  using FClone = Mutator::FClone;
  using FAsString = Mutator::FAsString;
  /*! \brief The packed function to the `InitializeWithTuneContext` function. */
  FInitializeWithTuneContext f_initialize_with_tune_context;
  /*! \brief The packed function to the `Apply` function. */
  FApply f_apply;
  /*! \brief The packed function to the `Clone` function. */
  FClone f_clone;
  /*! \brief The packed function to the `AsString` function. */
  FAsString f_as_string;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_initialize_with_tune_context` is not visited
    // `f_apply` is not visited
    // `f_clone` is not visited
    // `f_as_string` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& context) final;
  Optional<tir::Trace> Apply(const tir::Trace& trace,
                             support::LinearCongruentialEngine::TRandState* rand_state) final;
  Mutator Clone() const final;

  static constexpr const char* _type_key = "meta_schedule.PyMutator";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyMutatorNode, MutatorNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_MUTATOR_H_
