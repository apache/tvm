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

#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace meta_schedule {

class TuneContext;
class Postproc;

/*!
 * \brief Rules to apply a postprocessor to a schedule.
 */
class PostprocNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~PostprocNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Initialize the design space generator with tuning context.
   * \param context The tuning context for initialization.
   * \note This method is supposed to be called only once before every other method.
   */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Apply a postprocessor to the given schedule.
   * \param sch The schedule to be post processed.
   * \return Whether the postprocessor was successfully applied.
   */
  virtual bool Apply(const tir::Schedule& sch) = 0;

  /*!
   * \brief Clone the postprocessor.
   * \return The cloned postprocessor.
   */
  virtual Postproc Clone() const = 0;

  static constexpr const char* _type_key = "meta_schedule.Postproc";
  TVM_DECLARE_BASE_OBJECT_INFO(PostprocNode, Object);
};

/*!
 * \brief Managed reference to PostprocNode
 * \sa PostprocNode
 */
class Postproc : public runtime::ObjectRef {
 public:
  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  /*!
   * \brief Apply a postprocessor to the given schedule.
   * \param sch The schedule to be post processed.
   * \return Whether the postprocessor was successfully applied.
   */
  using FApply = runtime::TypedPackedFunc<bool(const tir::Schedule&)>;
  /*!
   * \brief Clone the postprocessor.
   * \return The cloned postprocessor.
   */
  using FClone = runtime::TypedPackedFunc<Postproc()>;
  /*!
   * \brief Get the postprocessor function as string with name.
   * \return The string of the postprocessor function.
   */
  using FAsString = runtime::TypedPackedFunc<String()>;
  /*!
   * \brief Create a postprocessor with customized methods on the python-side.
   * \param f_initialize_with_tune_context The packed function of `InitializeWithTuneContext`.
   * \param f_apply The packed function of `Apply`.
   * \param f_clone The packed function of `Clone`.
   * \param f_as_string The packed function of `AsString`.
   * \return The postprocessor created.
   */
  TVM_DLL static Postproc PyPostproc(FInitializeWithTuneContext f_initialize_with_tune_context,  //
                                     FApply f_apply,                                             //
                                     FClone f_clone,                                             //
                                     FAsString f_as_string);
  /*!
   * \brief Create a postprocessor that checks if all loops are static
   * \return The postprocessor created
   */
  TVM_DLL static Postproc DisallowDynamicLoop();
  /*!
   * \brief Create a postprocessor that rewrites the cooperative fetch annotation to
   * actual vectorized cooperative fetching in loop bindings.
   * \return The postprocessor created.
   */
  TVM_DLL static Postproc RewriteCooperativeFetch();
  /*!
   * \brief Creates a postprocessor that applies parallelization, vectorization and auto unrolling
   * according to the annotation of each block
   * \return The postprocessor created
   */
  TVM_DLL static Postproc RewriteParallelVectorizeUnroll();
  /*!
   * \brief Create a postprocessor that rewrites reduction block by moving the init block out.
   * \return The postprocessor created.
   */
  TVM_DLL static Postproc RewriteReductionBlock();
  /*!
   * \brief Create a postprocessor that adds thread binding to unbound blocks
   * \param max_threadblocks The max number of threadblocks in the cuda device.
   * \return The postprocessor created.
   */
  TVM_DLL static Postproc RewriteUnboundBlock(int max_threadblocks);
  /*!
   * \brief Create a postprocessor that applies tensorization to annotated blocks
   * \param vectorize_init_loop Whether or not vectorize the initialization loop produced by
   * DecomposeReduction
   * \return The postprocessor created.
   */
  TVM_DLL static Postproc RewriteTensorize(bool vectorize_init_loop = false);
  /*!
   * \brief Creates a postprocessor that verifies if the GPU code is correct
   * \return The postprocessor created
   */
  TVM_DLL static Postproc VerifyGPUCode();
  /*!
   * \brief Creates a postprocessor that rewrites the layout of input tensor
   * \note Weight layout rewrite is supported so far, activation layout rewrite will be added.
   * \return The postprocessor created
   */
  TVM_DLL static Postproc RewriteLayout();
  /*! \brief Create default postprocessors for LLVM */
  TVM_DLL static Array<Postproc, void> DefaultLLVM();
  /*! \brief Create default postprocessors for x86 VNNI */
  TVM_DLL static Array<Postproc, void> DefaultVNNI();
  /*! \brief Create default postprocessors for CUDA */
  TVM_DLL static Array<Postproc, void> DefaultCUDA();
  /*! \brief Create default postprocessors for CUDA with TensorCore */
  TVM_DLL static Array<Postproc, void> DefaultCUDATensorCore();
  /*! \brief Create default postprocessors for Hexagon */
  TVM_DLL static Array<Postproc, void> DefaultHexagon();

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Postproc, ObjectRef, PostprocNode);
};

/*! \brief The postprocessor with customized methods on the python-side. */
class PyPostprocNode : public PostprocNode {
 public:
  using FInitializeWithTuneContext = Postproc::FInitializeWithTuneContext;
  using FApply = Postproc::FApply;
  using FClone = Postproc::FClone;
  using FAsString = Postproc::FAsString;
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
  bool Apply(const tir::Schedule& sch) final;
  Postproc Clone() const final;

  static constexpr const char* _type_key = "meta_schedule.PyPostproc";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyPostprocNode, PostprocNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_POSTPROC_H_
