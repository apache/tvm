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
#ifndef TVM_META_SCHEDULE_PROFILER_H_
#define TVM_META_SCHEDULE_PROFILER_H_

#include <tvm/ir/module.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/target/target.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace meta_schedule {

class ScopedTimer {
 public:
  ~ScopedTimer() {
    if (deferred_ != nullptr) {
      deferred_();
    }
  }

 private:
  friend class Profiler;

  explicit ScopedTimer(runtime::TypedPackedFunc<void()> deferred) : deferred_(deferred) {}
  runtime::TypedPackedFunc<void()> deferred_;
};

/*! \brief A generic profiler */
class ProfilerNode : public runtime::Object {
 public:
  /*! \brief The segments that are already profiled */
  std::unordered_map<std::string, double> stats_sec;
  /*! \brief Counter for the total time used */
  runtime::PackedFunc total_timer;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `stats_sec` is not visited.
    // `total_timer` is not visited.
  }

  static constexpr const char* _type_key = "meta_schedule.Profiler";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProfilerNode, runtime::Object);

 public:
  /*! \brief Get the internal stats of the running time */
  Map<String, FloatImm> Get() const;
  /*! \brief Return a summary of profiling results as table format */
  String Table() const;
};

/*!
 * \brief Managed reference to ProfilerNode
 * \sa ProfilerNode
 */
class Profiler : public runtime::ObjectRef {
 public:
  Profiler();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Profiler, runtime::ObjectRef, ProfilerNode);

  /*! \brief Entering the scope of the context manager */
  void EnterWithScope();
  /*! \brief Exiting the scope of the context manager */
  void ExitWithScope();
  /*! \brief Returns the current profiler */
  static Optional<Profiler> Current();
  /*!
   * \brief Profile the time usage in the given scope in the given name.
   * \param name Name for the scope.
   * \return A scope timer for time profiling.
   */
  static ScopedTimer TimedScope(String name);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_PROFILER_H_
