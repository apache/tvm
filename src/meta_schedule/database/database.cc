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
#include <tvm/ffi/reflection/registry.h>

#include "../module_equality.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/******** Workload ********/

Workload::Workload(IRModule mod) {
  ObjectPtr<WorkloadNode> n = ffi::make_object<WorkloadNode>();
  n->mod = mod;
  n->shash = ModuleEquality::Create("structural")->Hash(mod);
  data_ = std::move(n);
}

Workload::Workload(IRModule mod, Workload::THashCode shash) {
  ObjectPtr<WorkloadNode> n = ffi::make_object<WorkloadNode>();
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
  return ffi::Array<ffi::Any>{SHash2Str(this->shash), ffi::String(b64_mod)};
}

Workload Workload::FromJSON(const ObjectRef& json_obj) {
  IRModule mod{ffi::UnsafeInit()};
  THashCode shash = 0;
  try {
    const ffi::ArrayObj* json_array = json_obj.as<ffi::ArrayObj>();
    CHECK(json_array && json_array->size() == 2);
    // Load json[0] => shash
    ffi::String str_shash = json_array->at(0).cast<ffi::String>();
    // Load json[1] => mod
    {
      ffi::String b64_mod = json_array->at(1).cast<ffi::String>();
      std::string json_mod = Base64Decode(b64_mod);
      mod = LoadJSON(json_mod).cast<IRModule>();
      std::stringstream(str_shash) >> shash;
    }
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
               << "\nThe error is: " << e.what();
  }
  return Workload(mod, shash);
}

/******** TuningRecord ********/

TuningRecord::TuningRecord(tir::Trace trace, Workload workload,
                           ffi::Optional<ffi::Array<FloatImm>> run_secs,
                           ffi::Optional<Target> target,
                           ffi::Optional<ffi::Array<ArgInfo>> args_info) {
  ObjectPtr<TuningRecordNode> n = ffi::make_object<TuningRecordNode>();
  n->trace = trace;
  n->workload = workload;
  n->run_secs = run_secs;
  n->target = target;
  n->args_info = args_info;
  this->data_ = n;
}

bool WorkloadEqual::operator()(const Workload& a, const Workload& b) const {
  return a->shash == b->shash && mod_eq_.Equal(a->mod, b->mod);
}

MeasureCandidate TuningRecordNode::AsMeasureCandidate() const {
  tir::Schedule sch =
      tir::Schedule::Traced(workload->mod, -1, 0, tir::ScheduleErrorRenderLevel::kDetail);
  trace->ApplyToSchedule(sch, false, nullptr);
  return MeasureCandidate(sch, ArgInfo::FromEntryFunc(sch->mod(), /*remove_preproc=*/true));
}

ObjectRef TuningRecordNode::AsJSON() const {
  ffi::Optional<ffi::Array<ObjectRef>> json_args_info;
  ffi::Optional<ObjectRef> json_target;
  if (args_info.defined()) {
    ffi::Array<ObjectRef> info;
    info.reserve(args_info.value().size());
    for (const ArgInfo& arg_info : args_info.value()) {
      info.push_back(arg_info->AsJSON());
    }
    json_args_info = info;
  }
  if (target.defined()) {
    json_target = target.value()->Export();
  }
  return ffi::Array<ObjectRef>{trace->AsJSON(false),  //
                               run_secs,              //
                               json_target,           //
                               json_args_info};
}

bool TuningRecordNode::IsValid() const {
  if (!GetNumValidInstructions(trace->insts, /*remove_postproc*/ true)) {
    return false;
  }
  if (run_secs.defined()) {
    for (const auto& run_sec : run_secs.value()) {
      // kMaxMeanTime(1e10) is used as a stub for undefined measurement times.
      if (run_sec.defined() && run_sec->value != SortTuningRecordByMeanRunSecs::kMaxMeanTime) {
        return true;
      }
    }
  }
  return false;
}

TuningRecord TuningRecord::FromJSON(const ObjectRef& json_obj, const Workload& workload) {
  tir::Trace trace{ffi::UnsafeInit()};
  ffi::Optional<ffi::Array<FloatImm>> run_secs;
  ffi::Optional<Target> target;
  ffi::Optional<ffi::Array<ArgInfo>> args_info;
  try {
    const ffi::ArrayObj* json_array = json_obj.as<ffi::ArrayObj>();
    CHECK(json_array && json_array->size() == 4);
    // Load json[1] => run_secs
    if (json_array->at(1) != nullptr) {
      run_secs = AsFloatArray(json_array->at(1).cast<ObjectRef>());
    }
    // Load json[2] => target
    if (json_array->at(2) != nullptr) {
      target = Target(json_array->at(2).cast<ffi::Map<ffi::String, ffi::Any>>());
    }
    // Load json[3] => args_info
    if (json_array->at(3) != nullptr) {
      const ffi::ArrayObj* json_args_info = json_array->at(3).cast<const ffi::ArrayObj*>();
      ffi::Array<ArgInfo> info;
      info.reserve(json_args_info->size());
      for (Any json_arg_info : *json_args_info) {
        info.push_back(ArgInfo::FromJSON(json_arg_info.cast<ObjectRef>()));
      }
      args_info = info;
    }
    // Load json[0] => trace
    {
      auto json_trace = json_array->at(0).cast<ObjectRef>();
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
  return TuningRecord(trace, workload, run_secs, target, args_info);
}

/******** Database ********/
DatabaseNode::DatabaseNode(ffi::String mod_eq_name) {
  mod_eq_ = ModuleEquality::Create(mod_eq_name);
}
DatabaseNode::~DatabaseNode() = default;

ffi::Optional<TuningRecord> DatabaseNode::QueryTuningRecord(const IRModule& mod,
                                                            const Target& target,
                                                            const ffi::String& workload_name) {
  if (!this->HasWorkload(mod)) {
    return std::nullopt;
  }
  ffi::Array<TuningRecord> records = this->GetTopK(this->CommitWorkload(mod), 1);
  if (records.empty()) {
    return std::nullopt;
  }
  ICHECK_EQ(records.size(), 1);
  return records[0];
}

ffi::Optional<tir::Schedule> DatabaseNode::QuerySchedule(const IRModule& mod, const Target& target,
                                                         const ffi::String& workload_name) {
  if (ffi::Optional<TuningRecord> opt_record =
          this->QueryTuningRecord(mod, target, workload_name)) {
    TuningRecord record = opt_record.value();
    tir::Schedule sch =
        tir::Schedule::Traced(record->workload->mod, /*seed=*/-1, /*debug_mask=*/0,
                              /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);
    record->trace->ApplyToSchedule(sch, false);
    return sch;
  } else {
    return std::nullopt;
  }
}

ffi::Optional<IRModule> DatabaseNode::QueryIRModule(const IRModule& mod, const Target& target,
                                                    const ffi::String& workload_name) {
  if (ffi::Optional<tir::Schedule> opt_sch = this->QuerySchedule(mod, target, workload_name)) {
    return opt_sch.value()->mod();
  } else {
    return std::nullopt;
  }
}

void DatabaseNode::DumpPruned(Database destination) {
  std::unordered_map<Workload, TuningRecord, ObjectPtrHash, ObjectPtrEqual> workload2record;
  for (const TuningRecord& record : this->GetAllTuningRecords()) {
    if (record->IsValid()) {
      auto it = workload2record.find(record->workload);
      if (it == workload2record.end()) {
        workload2record.insert({record->workload, record});
      } else if (SortTuningRecordByMeanRunSecs()(record, it->second)) {
        it->second = record;
      }
    }
  }
  for (auto& kv : workload2record) {
    Workload workload = kv.first;
    TuningRecord record = kv.second;
    workload = destination->CommitWorkload(workload->mod);
    destination->CommitTuningRecord(TuningRecord(/*trace=*/record->trace, /*workload=*/workload,
                                                 /*run_secs=*/record->run_secs,
                                                 /*target=*/record->target,
                                                 /*args_info=*/record->args_info));
  }
}

std::vector<Database>* ThreadLocalDatabases() {
  static thread_local std::vector<Database> tls;
  return &tls;
}

void Database::EnterWithScope() { ThreadLocalDatabases()->push_back(*this); }

void Database::ExitWithScope() { ThreadLocalDatabases()->pop_back(); }

ffi::Optional<Database> Database::Current() {
  std::vector<Database>* tls = ThreadLocalDatabases();
  if (tls->empty()) {
    return std::nullopt;
  } else {
    return tls->back();
  }
}

/******** PyDatabase ********/
PyDatabaseNode::PyDatabaseNode(ffi::String mod_eq_name) : DatabaseNode(mod_eq_name) {}

Database Database::PyDatabase(PyDatabaseNode::FHasWorkload f_has_workload,
                              PyDatabaseNode::FCommitWorkload f_commit_workload,
                              PyDatabaseNode::FCommitTuningRecord f_commit_tuning_record,
                              PyDatabaseNode::FGetTopK f_get_top_k,
                              PyDatabaseNode::FGetAllTuningRecords f_get_all_tuning_records,
                              PyDatabaseNode::FQueryTuningRecord f_query_tuning_record,
                              PyDatabaseNode::FQuerySchedule f_query_schedule,
                              PyDatabaseNode::FQueryIRModule f_query_ir_module,
                              PyDatabaseNode::FSize f_size, ffi::String mod_eq_name) {
  ObjectPtr<PyDatabaseNode> n = ffi::make_object<PyDatabaseNode>(mod_eq_name);
  n->f_has_workload = f_has_workload;
  n->f_commit_workload = f_commit_workload;
  n->f_commit_tuning_record = f_commit_tuning_record;
  n->f_get_top_k = f_get_top_k;
  n->f_get_all_tuning_records = f_get_all_tuning_records;
  n->f_query_tuning_record = f_query_tuning_record;
  n->f_query_schedule = f_query_schedule;
  n->f_query_ir_module = f_query_ir_module;
  n->f_size = f_size;
  return Database(n);
}

/******** FFI ********/

TVM_FFI_STATIC_INIT_BLOCK() {
  WorkloadNode::RegisterReflection();
  TuningRecordNode::RegisterReflection();
  PyDatabaseNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("meta_schedule.Workload", [](IRModule mod) { return Workload(mod); })
      .def_method("meta_schedule.WorkloadAsJSON", &WorkloadNode::AsJSON)
      .def("meta_schedule.WorkloadFromJSON", &Workload::FromJSON)
      .def("meta_schedule.TuningRecord",
           [](tir::Trace trace, Workload workload, ffi::Optional<ffi::Array<FloatImm>> run_secs,
              ffi::Optional<Target> target, ffi::Optional<ffi::Array<ArgInfo>> args_info) {
             return TuningRecord(trace, workload, run_secs, target, args_info);
           })
      .def_method("meta_schedule.TuningRecordAsMeasureCandidate",
                  &TuningRecordNode::AsMeasureCandidate)
      .def_method("meta_schedule.TuningRecordAsJSON", &TuningRecordNode::AsJSON)
      .def("meta_schedule.TuningRecordFromJSON", TuningRecord::FromJSON)
      .def_method("meta_schedule.DatabaseEnterWithScope", &Database::EnterWithScope)
      .def_method("meta_schedule.DatabaseExitWithScope", &Database::ExitWithScope)
      .def("meta_schedule.DatabaseCurrent", Database::Current)
      .def_method("meta_schedule.DatabaseHasWorkload", &DatabaseNode::HasWorkload)
      .def_method("meta_schedule.DatabaseCommitWorkload", &DatabaseNode::CommitWorkload)
      .def_method("meta_schedule.DatabaseCommitTuningRecord", &DatabaseNode::CommitTuningRecord)
      .def_method("meta_schedule.DatabaseGetTopK", &DatabaseNode::GetTopK)
      .def_method("meta_schedule.DatabaseGetAllTuningRecords", &DatabaseNode::GetAllTuningRecords)
      .def_method("meta_schedule.DatabaseSize", &DatabaseNode::Size)
      .def_method("meta_schedule.DatabaseQueryTuningRecord", &DatabaseNode::QueryTuningRecord)
      .def_method("meta_schedule.DatabaseQuerySchedule", &DatabaseNode::QuerySchedule)
      .def_method("meta_schedule.DatabaseQueryIRModule", &DatabaseNode::QueryIRModule)
      .def_method("meta_schedule.DatabaseDumpPruned", &DatabaseNode::DumpPruned)
      .def("meta_schedule.DatabasePyDatabase", Database::PyDatabase);
}

}  // namespace meta_schedule
}  // namespace tvm
