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
#ifndef TVM_META_SCHEDULE_APPLY_HISTORY_BEST_H_
#define TVM_META_SCHEDULE_APPLY_HISTORY_BEST_H_

#include <tvm/ir/module.h>
#include <tvm/meta_schedule/database.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/target/target.h>
#include <tvm/te/tensor.h>

namespace tvm {
namespace meta_schedule {

/*!
 * \brief An integration context that allows application of historically best records from a
 * database
 */
class ApplyHistoryBestNode : public runtime::Object {
 public:
  /*! \brief A callback function that filters TE compute */
  using FTEFilterFunc =
      runtime::TypedPackedFunc<Optional<tir::PrimFunc>(const Array<te::Tensor, void>&)>;
  /*! \brief  A callback function that takes a tuning record and does something with it */
  using FTakeTuningRecord = runtime::TypedPackedFunc<void(const TuningRecord&)>;
  using FDirectDispatch = runtime::TypedPackedFunc<Optional<IRModule>(const IRModule&)>;

  /*! \brief The database to be queried from */
  Database database{nullptr};
  /*! \brief The filtering function for TE computation */
  FTEFilterFunc te_filter_func{nullptr};
  /*! \brief The logging function to be used */
  PackedFunc logging_func;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("database", &database);
    // `te_filter_func` is not visited
    // `logging_func` is not visited
  }
  /*!
   * \brief Query the best entry from the database
   * \param task_name The name of the task to be queried
   * \param mod The module to be queried
   * \param target The target to be queried
   * \param dispatched The IRs after dispatch
   * \param f_take_tuning_record A callback function that takes a tuning record and does something
   *   with it.
   * \param f_direct_dispatch A function that directly dispatches an IRModule to the given workload
   *   as result if available, skipping the database query.
   */
  Optional<IRModule> Query(runtime::String task_name, IRModule mod, Target target,
                           Optional<Array<IRModule>> dispatched,
                           FTakeTuningRecord f_take_tuning_record,
                           FDirectDispatch f_direct_dispatch = nullptr);

  static constexpr const char* _type_key = "meta_schedule.ApplyHistoryBest";
  TVM_DECLARE_FINAL_OBJECT_INFO(ApplyHistoryBestNode, runtime::Object);
};

/*!
 * \brief Managed reference to ApplyHistoryBestNode
 * \sa ApplyHistoryBestNode
 */
class ApplyHistoryBest : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param database The database to be queried from
   * \param te_filter_func The filtering function for TE computation
   * \param logging_func The logging function to use
   */
  explicit ApplyHistoryBest(Database database, ApplyHistoryBestNode::FTEFilterFunc te_filter_func,
                            PackedFunc logging_func);
  /*!
   * \brief The current ApplyHistoryBest in the context
   * \return The ApplyHistoryBest in the current scope.
   */
  static Optional<ApplyHistoryBest> Current();

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ApplyHistoryBest, runtime::ObjectRef,
                                                    ApplyHistoryBestNode);

 protected:
  friend class ApplyHistoryBestInternal;
  /*! \brief Entering the scope of the context manager */
  void EnterWithScope();
  /*! \brief Exiting the scope of the context manager */
  void ExitWithScope();
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_APPLY_HISTORY_BEST_H_
