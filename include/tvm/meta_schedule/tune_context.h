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
#ifndef TVM_META_SCHEDULE_TUNE_CONTEXT_H_
#define TVM_META_SCHEDULE_TUNE_CONTEXT_H_

#include <tvm/ir/module.h>
#include <tvm/meta_schedule/space_generator.h>
#include <tvm/support/random_engine.h>
#include <tvm/target/target.h>

namespace tvm {
namespace meta_schedule {

/*! \brief The auto tuning context. */
class TuneContextNode : public runtime::Object {
 public:
  /*! \brief The workload to be tuned. */
  Optional<IRModule> mod;
  /*! \brief The target to be tuned for. */
  Optional<Target> target;
  /*! \brief The design space generator. */
  Optional<SpaceGenerator> space_generator;
  /*! \brief The name of the tuning task. */
  Optional<String> task_name;
  /*! \brief The random state. */
  support::LinearCongruentialEngine::TRandState rand_state;
  /*! \brief The number of threads to be used. */
  int num_threads;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("mod", &mod);
    v->Visit("target", &target);
    v->Visit("space_generator", &space_generator);
    v->Visit("task_name", &task_name);
    v->Visit("rand_state", &rand_state);
    v->Visit("num_threads", &num_threads);
  }

  static constexpr const char* _type_key = "meta_schedule.TuneContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuneContextNode, Object);
};

/*!
 * \brief Managed reference to TuneContextNode.
 * \sa TuneContextNode
 */
class TuneContext : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor.
   * \param mod The workload to be tuned.
   * \param target The target to be tuned for.
   * \param space_generator The design space generator.
   * \param task_name The name of the tuning task.
   * \param rand_state The random state.
   * \param num_threads The number of threads to be used.
   */
  TVM_DLL explicit TuneContext(Optional<IRModule> mod,                                    //
                               Optional<Target> target,                                   //
                               Optional<SpaceGenerator> space_generator,                  //
                               Optional<String> task_name,                                //
                               support::LinearCongruentialEngine::TRandState rand_state,  //
                               int num_threads);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TuneContext, ObjectRef, TuneContextNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_TUNE_CONTEXT_H_
