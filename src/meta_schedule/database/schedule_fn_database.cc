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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

class ScheduleFnDatabaseNode : public DatabaseNode {
 public:
  explicit ScheduleFnDatabaseNode(String mod_eq_name = "structural") : DatabaseNode(mod_eq_name) {}

  runtime::TypedPackedFunc<bool(tir::Schedule)> schedule_fn;

  void VisitAttrs(AttrVisitor* v) {
    // `schedule_fn` is not visited.
  }

  static constexpr const char* _type_key = "meta_schedule.ScheduleFnDatabase";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleFnDatabaseNode, DatabaseNode);

 public:
  Optional<TuningRecord> QueryTuningRecord(const IRModule& mod, const Target& target,
                                           const String& workload_name) final {
    if (Optional<tir::Schedule> sch = this->QuerySchedule(mod, target, workload_name)) {
      return TuningRecord(sch.value()->trace().value(),
                          /*workload=*/Workload(mod, 0),  //
                          /*run_secs=*/NullOpt,           //
                          /*target=*/target,              //
                          /*arg_info=*/NullOpt);
    }
    return NullOpt;
  }

  Optional<tir::Schedule> QuerySchedule(const IRModule& mod, const Target& target,
                                        const String& workload_name) final {
    tir::Schedule sch =
        tir::Schedule::Traced(WithAttr<IRModule>(mod, "task_name", workload_name),
                              /*rand_state=*/-1,
                              /*debug_mode=*/0,
                              /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);
    if (!schedule_fn(sch)) {
      return NullOpt;
    }
    return sch;
  }

  bool HasWorkload(const IRModule& mod) final {
    LOG(FATAL) << "NotImplementedError: ScheduleFnDatabase.HasWorkload";
    throw;
  }

  Workload CommitWorkload(const IRModule& mod) final {
    LOG(FATAL) << "NotImplementedError: ScheduleFnDatabase.CommitWorkload";
    throw;
  }

  void CommitTuningRecord(const TuningRecord& record) final {
    LOG(FATAL) << "NotImplementedError: ScheduleFnDatabase.CommitTuningRecord";
    throw;
  }

  Array<TuningRecord> GetTopK(const Workload& workload, int top_k) final {
    LOG(FATAL) << "NotImplementedError: ScheduleFnDatabase.GetTopK";
    throw;
  }

  Array<TuningRecord> GetAllTuningRecords() final {
    LOG(FATAL) << "NotImplementedError: ScheduleFnDatabase.GetAllTuningRecords";
    throw;
  }

  int64_t Size() final {
    LOG(FATAL) << "NotImplementedError: ScheduleFnDatabase.size";
    throw;
  }
};

Database Database::ScheduleFnDatabase(runtime::TypedPackedFunc<bool(tir::Schedule)> schedule_fn,
                                      String mod_eq_name) {
  ObjectPtr<ScheduleFnDatabaseNode> n = make_object<ScheduleFnDatabaseNode>(mod_eq_name);
  n->schedule_fn = std::move(schedule_fn);
  return Database(n);
}

TVM_REGISTER_NODE_TYPE(ScheduleFnDatabaseNode);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseScheduleFnDatabase")
    .set_body_typed(Database::ScheduleFnDatabase);

}  // namespace meta_schedule
}  // namespace tvm
