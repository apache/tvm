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

/******** Workload ********/

Workload::Workload(IRModule mod) {
  ObjectPtr<WorkloadNode> n = runtime::make_object<WorkloadNode>();
  n->shash = tvm::StructuralHash()(mod);
  n->mod = mod;
  data_ = std::move(n);
}

Workload::Workload(IRModule mod, Workload::THashCode shash) {
  ObjectPtr<WorkloadNode> n = runtime::make_object<WorkloadNode>();
  n->mod = mod;
  n->shash = shash;
  data_ = std::move(n);
}

ObjectRef WorkloadNode::AsJSON() const {
  // Convert `this->mod` to JSON
  std::string json_mod = tvm::SaveJSON(this->mod);
  // Dump the JSON string to base64
  std::string b64_mod = Base64Encode(json_mod);
  // Output
  return Array<ObjectRef>{SHash2Str(this->shash), String(b64_mod)};
}

Workload Workload::FromJSON(const ObjectRef& json_obj) {
  IRModule mod{nullptr};
  THashCode shash = 0;
  try {
    const ArrayNode* json_array = json_obj.as<ArrayNode>();
    CHECK(json_array && json_array->size() == 2);
    // Load json[0] => shash
    String str_shash = Downcast<String>(json_array->at(0));
    // Load json[1] => mod
    {
      String b64_mod = Downcast<String>(json_array->at(1));
      std::string json_mod = Base64Decode(b64_mod);
      mod = Downcast<IRModule>(LoadJSON(json_mod));
    }
    // Verify SHash(mod) == shash
    shash = tvm::StructuralHash()(mod);
    String recalc_shash = SHash2Str(shash);
    CHECK_EQ(recalc_shash, str_shash) << "ValueError: Structural hash changed. Given: " << str_shash
                                      << "; Recalculated: " << recalc_shash;
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
               << "\nThe error is: " << e.what();
  }
  return Workload(mod, shash);
}

/******** TuningRecord ********/

TuningRecord::TuningRecord(tir::Trace trace, Array<FloatImm> run_secs, Workload workload,
                           Target target, Array<ArgInfo> args_info) {
  ObjectPtr<TuningRecordNode> n = make_object<TuningRecordNode>();
  n->trace = trace;
  n->run_secs = run_secs;
  n->workload = workload;
  n->target = target;
  n->args_info = args_info;
  this->data_ = n;
}

ObjectRef TuningRecordNode::AsJSON() const {
  Array<ObjectRef> json_args_info;
  json_args_info.reserve(args_info.size());
  for (const ArgInfo& arg_info : args_info) {
    json_args_info.push_back(arg_info->AsJSON());
  }
  return Array<ObjectRef>{trace->AsJSON(false),  //
                          run_secs,              //
                          target->Export(),      //
                          json_args_info};
}

TuningRecord TuningRecord::FromJSON(const ObjectRef& json_obj, const Workload& workload) {
  tir::Trace trace{nullptr};
  Array<FloatImm> run_secs{nullptr};
  Target target{nullptr};
  Array<ArgInfo> args_info;
  try {
    const ArrayNode* json_array = json_obj.as<ArrayNode>();
    CHECK(json_array && json_array->size() == 4);
    // Load json[1] => run_secs
    run_secs = Downcast<Array<FloatImm>>(json_array->at(1));
    // Load json[2] => target
    target = Target(Downcast<Map<String, ObjectRef>>(json_array->at(2)));
    // Load json[3] => args_info
    {
      const ArrayNode* json_args_info = json_array->at(3).as<ArrayNode>();
      args_info.reserve(json_args_info->size());
      for (const ObjectRef& json_arg_info : *json_args_info) {
        args_info.push_back(ArgInfo::FromJSON(json_arg_info));
      }
    }
    // Load json[0] => trace
    {
      const ObjectRef& json_trace = json_array->at(0);
      tir::Schedule sch =
          tir::Schedule::Traced(workload->mod, /*seed=*/-1, /*debug_mask=*/0,
                                /*error_render_level=*/tir::ScheduleErrorRenderLevel::kNone);
      tir::Trace::ApplyJSONToSchedule(json_trace, sch);
      trace = sch->trace().value();
    }
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
               << "\nThe error is: " << e.what();
  }
  return TuningRecord(trace, run_secs, workload, target, args_info);
}

/******** PyDatabase ********/

Database Database::PyDatabase(PyDatabaseNode::FHasWorkload f_has_workload,
                              PyDatabaseNode::FCommitWorkload f_commit_workload,
                              PyDatabaseNode::FCommitTuningRecord f_commit_tuning_record,
                              PyDatabaseNode::FGetTopK f_get_top_k, PyDatabaseNode::FSize f_size) {
  ObjectPtr<PyDatabaseNode> n = make_object<PyDatabaseNode>();
  n->f_has_workload = f_has_workload;
  n->f_commit_workload = f_commit_workload;
  n->f_commit_tuning_record = f_commit_tuning_record;
  n->f_get_top_k = f_get_top_k;
  n->f_size = f_size;
  return Database(n);
}

/******** FFI ********/

TVM_REGISTER_NODE_TYPE(WorkloadNode);
TVM_REGISTER_NODE_TYPE(TuningRecordNode);
TVM_REGISTER_OBJECT_TYPE(DatabaseNode);
TVM_REGISTER_NODE_TYPE(PyDatabaseNode);
TVM_REGISTER_GLOBAL("meta_schedule.Workload").set_body_typed([](IRModule mod) {
  return Workload(mod);
});
TVM_REGISTER_GLOBAL("meta_schedule.WorkloadAsJSON")
    .set_body_method<Workload>(&WorkloadNode::AsJSON);
TVM_REGISTER_GLOBAL("meta_schedule.WorkloadFromJSON").set_body_typed(&Workload::FromJSON);
TVM_REGISTER_GLOBAL("meta_schedule.TuningRecord")
    .set_body_typed([](tir::Trace trace, Array<FloatImm> run_secs, Workload workload, Target target,
                       Array<ArgInfo> args_info) {
      return TuningRecord(trace, run_secs, workload, target, args_info);
    });
TVM_REGISTER_GLOBAL("meta_schedule.TuningRecordAsJSON")
    .set_body_method<TuningRecord>(&TuningRecordNode::AsJSON);
TVM_REGISTER_GLOBAL("meta_schedule.TuningRecordFromJSON").set_body_typed(TuningRecord::FromJSON);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseHasWorkload")
    .set_body_method<Database>(&DatabaseNode::HasWorkload);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseCommitWorkload")
    .set_body_method<Database>(&DatabaseNode::CommitWorkload);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseCommitTuningRecord")
    .set_body_method<Database>(&DatabaseNode::CommitTuningRecord);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseGetTopK")
    .set_body_method<Database>(&DatabaseNode::GetTopK);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseSize").set_body_method<Database>(&DatabaseNode::Size);
TVM_REGISTER_GLOBAL("meta_schedule.DatabasePyDatabase").set_body_typed(Database::PyDatabase);

}  // namespace meta_schedule
}  // namespace tvm
