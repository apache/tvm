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

#include <utility>
#include <vector>

namespace tvm {
namespace meta_schedule {

struct ScopedTimer {
  std::function<void()> func;
  explicit ScopedTimer(std::function<void()> func) : func(func) {}
  ~ScopedTimer() { func(); }
};

/*!
 * \brief A profiler to count tuning time cost in different parts.
 */
class ProfilerNode : public runtime::Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("stats", &stats); }

  /*!
   * \brief Profile the time usage in the given scope in the given name.
   * \param name Name for the scope.
   * \return A scope timer for time profiling.
   */
  static ScopedTimer TimeScope(String name);

  /*!
   * \brief Get the profiling results.
   * \return The tuning profiling results as a dict.
   */
  Map<String, FloatImm> Get() const { return stats; }

  /*!
   * \brief Start the timer for a new context.
   * \param name Name of the context.
   */
  void StartContextTimer(String name);

  /*! \brief End the timer for the most recent context. */
  void EndContextTimer();

  static constexpr const char* _type_key = "meta_schedule.Profiler";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProfilerNode, runtime::Object);

 protected:
  Map<String, FloatImm> stats;
  std::vector<std::pair<String, std::chrono::time_point<std::chrono::high_resolution_clock>>> stack;
};

/*!
 * \brief Managed reference to ProfilerNode
 * \sa ProfilerNode
 */
class Profiler : public runtime::ObjectRef {
 public:
  Profiler();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Profiler, runtime::ObjectRef, ProfilerNode);

 protected:
  friend class ProfilerInternal;

  /*! \brief Entering the scope of the context manager */
  void EnterWithScope();
  /*! \brief Exiting the scope of the context manager */
  void ExitWithScope();
};

struct ProfilerThreadLocalEntry {
  Optional<Profiler> ctx;
};
using ProfilerThreadLocalStore = dmlc::ThreadLocalStore<ProfilerThreadLocalEntry>;

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_PROFILER_H_
