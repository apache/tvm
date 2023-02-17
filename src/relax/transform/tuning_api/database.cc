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

/*!
 * \file src/relax/transform/tuning_api/database.cc
 * \brief Database of tuning APIs.
 */
#include <tvm/relax/tuning_api.h>

#include <set>
#include <thread>
#include <unordered_map>

#include "../../../meta_schedule/utils.h"

namespace tvm {
namespace meta_schedule {

void JSONFileAppendLine(const String& path, const std::string& line);
std::vector<ObjectRef> JSONFileReadLines(const String& path, int num_threads, bool allow_missing);

}  // namespace meta_schedule
}  // namespace tvm

namespace tvm {
namespace relax {

TuningRecord::TuningRecord(Trace trace, Optional<Array<FloatImm>> run_secs) {
  ObjectPtr<TuningRecordNode> n = make_object<TuningRecordNode>();
  n->trace = trace;
  n->run_secs = run_secs;
  this->data_ = n;
}

ObjectRef TuningRecordNode::AsJSON(bool include_irmod) const {
  return Array<ObjectRef>{trace->AsJSON(include_irmod),  //
                          run_secs};
}

TuningRecord TuningRecord::FromJSON(const ObjectRef& json_obj) {
  Trace trace{nullptr};
  Optional<Array<FloatImm>> run_secs{nullptr};
  try {
    const ArrayNode* json_array = json_obj.as<ArrayNode>();
    CHECK(json_array && json_array->size() == 2);
    // Load json[0] => trace
    {
      const ObjectRef& json_trace = json_array->at(0);
      trace = Trace::FromJSON(json_trace);
    }

    // Load json[1] => run_secs
    if (json_array->at(1).defined()) {
      run_secs = meta_schedule::AsFloatArray(json_array->at(1));
    }
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
               << "\nThe error is: " << e.what();
  }
  return TuningRecord(trace, run_secs);
}

/*! \brief The struct defining comparison function of sorting by mean run seconds. */
struct SortTuningRecordByMeanRunSecs {
  static const constexpr double kMaxMeanTime = 1e10;

  static double Mean(const Array<FloatImm>& a) {
    if (a.empty()) {
      return kMaxMeanTime;
    }
    double sum = 0.0;
    for (const FloatImm& i : a) {
      sum += i->value;
    }
    return sum / a.size();
  }

  bool operator()(const TuningRecord& a, const TuningRecord& b) const {
    double a_time = Mean(a->run_secs.value_or({}));
    double b_time = Mean(b->run_secs.value_or({}));
    return a_time < b_time;
  }
};

// TODO(tvm-team): Currently, we strictly treat each target separately.
// Since not every option in the target matters, this might be the overkill.
// Revisit this when we have better approach with target equality check.
inline std::string get_database_key(int workload_idx, Target target) {
  return std::to_string(workload_idx) + "/" + target->str();
}

/*! \brief The default database implementation, which mimics two database tables with two files.
 */
class JSONDatabaseNode : public DatabaseNode {
 public:
  /*! \brief The path to the workload table */
  String path_workload;
  /*! \brief The path to the tuning record table */
  String path_tuning_record;
  /*! \brief The path to the measurement table */
  String path_measurement_record;
  /*! \brief All the workloads in the database */
  std::unordered_map<meta_schedule::Workload, int, meta_schedule::WorkloadHash, WorkloadEqual>
      workloads2idx_;
  /*! \brief All the tuning records in the database */
  std::unordered_map<std::string, std::multiset<TuningRecord, SortTuningRecordByMeanRunSecs>>
      tuning_records_;

  /*! \brief Measurement logs in the database */
  std::unordered_map<std::string, Array<FloatImm>> measurement_records_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("path_workload", &path_workload);
    v->Visit("path_tuning_record", &path_tuning_record);
    v->Visit("path_measurement_record", &path_measurement_record);
    // `workloads2idx_` is not visited
    // `tuning_records_` is not visited
    // `measurement_records_` is not visited
  }

  static constexpr const char* _type_key = "relax.tuning_api.JSONDatabase";
  TVM_DECLARE_FINAL_OBJECT_INFO(JSONDatabaseNode, DatabaseNode);

 public:
  bool HasWorkload(const IRModule& mod) {
    return workloads2idx_.find(meta_schedule::Workload(mod, tvm::StructuralHash()(mod))) !=
           workloads2idx_.end();
  }

  bool HasMeasurementRecord(const meta_schedule::Workload& workload, const Target& target) {
    int workload_idx = this->workloads2idx_.at(workload);
    std::string key = get_database_key(workload_idx, target);
    return measurement_records_.count(key) > 0;
  }

  bool HasTuningRecord(const meta_schedule::Workload& workload, const Target& target) {
    int workload_idx = this->workloads2idx_.at(workload);
    std::string key = get_database_key(workload_idx, target);
    return tuning_records_.count(key) > 0;
  }

  meta_schedule::Workload CommitWorkload(const IRModule& mod) {
    // Try to insert `mod` into `workloads_`
    decltype(this->workloads2idx_)::iterator it;
    bool inserted = false;
    std::tie(it, inserted) =
        this->workloads2idx_.emplace(meta_schedule::Workload(mod, tvm::StructuralHash()(mod)), -1);
    meta_schedule::Workload workload = it->first;
    // If `mod` is new in `workloads2idx_`, append it to the workload file
    if (inserted) {
      it->second = static_cast<int>(this->workloads2idx_.size()) - 1;
      meta_schedule::JSONFileAppendLine(this->path_workload,
                                        meta_schedule::JSONDumps(workload->AsJSON()));
    }
    return it->first;
  }

  void CommitMeasurementRecord(const meta_schedule::Workload& workload, const Target& target,
                               const Array<FloatImm>& run_secs) {
    int workload_idx = this->workloads2idx_.at(workload);
    std::string key = get_database_key(workload_idx, target);

    if (measurement_records_[key].size() == 0) {
      measurement_records_[key] = run_secs;
      meta_schedule::JSONFileAppendLine(this->path_measurement_record,
                                        meta_schedule::JSONDumps(Array<ObjectRef>{
                                            Integer(workload_idx), target->Export(),
                                            run_secs  //
                                        }));
    } else {
      LOG(WARNING) << "Measurement record for " << key
                   << " already exists. Use the existing one instead.";
    }
  }

  void CommitTuningRecord(const meta_schedule::Workload& workload, const Target& target,
                          const TuningRecord& record) {
    int workload_idx = this->workloads2idx_.at(workload);
    // There may exist multiple tuning records (with different traces) for a single key pair.
    std::string key = get_database_key(workload_idx, target);
    this->tuning_records_[key].insert(record);

    meta_schedule::JSONFileAppendLine(
        this->path_tuning_record, meta_schedule::JSONDumps(Array<ObjectRef>{
                                      Integer(workload_idx), target->Export(), record->AsJSON()}));
  }

  Array<TuningRecord> GetTopK(const meta_schedule::Workload& workload, const Target& target,
                              int top_k) {
    CHECK_GE(top_k, 0) << "ValueError: top_k must be non-negative";
    if (top_k == 0) {
      return {};
    }
    Array<TuningRecord> results;
    results.reserve(top_k);
    int counter = 0;
    int idx = this->workloads2idx_.at(workload);
    std::string key = get_database_key(idx, target);
    for (const TuningRecord& record : this->tuning_records_[key]) {
      results.push_back(record);
      if (++counter == top_k) {
        break;
      }
    }

    return results;
  }

  Array<FloatImm> GetMeasurementRecord(const meta_schedule::Workload& workload,
                                       const Target target) {
    int workload_idx = this->workloads2idx_.at(workload);
    return this->measurement_records_[get_database_key(workload_idx, target)];
  }
};

Database Database::JSONDatabase(String path_workload, String path_tuning_record,
                                String path_measurement_record, bool allow_missing) {
  int num_threads = std::thread::hardware_concurrency();
  ObjectPtr<JSONDatabaseNode> n = make_object<JSONDatabaseNode>();
  // Load `n->workloads2idx_` from `path_workload`
  std::vector<meta_schedule::Workload> workloads;
  {
    std::vector<ObjectRef> json_objs =
        meta_schedule::JSONFileReadLines(path_workload, num_threads, allow_missing);
    int n_objs = json_objs.size();
    n->workloads2idx_.reserve(n_objs);
    workloads.reserve(n_objs);
    for (int i = 0; i < n_objs; ++i) {
      meta_schedule::Workload workload = meta_schedule::Workload::FromJSON(json_objs[i]);
      n->workloads2idx_.emplace(workload, i);
      workloads.push_back(workload);
    }
  }
  // Load `n->tuning_records_` from `path_tuning_record`
  {
    std::vector<ObjectRef> json_objs =
        meta_schedule::JSONFileReadLines(path_tuning_record, num_threads, allow_missing);

    std::vector<int> workload_idxs;
    std::vector<Target> targets;
    std::vector<TuningRecord> records;
    int size = json_objs.size();
    workload_idxs.resize(size, -1);
    targets.resize(size, Target{nullptr});
    records.resize(size, TuningRecord{nullptr});
    support::parallel_for_dynamic(
        0, json_objs.size(), num_threads, [&](int thread_id, int task_id) {
          const ObjectRef& json_obj = json_objs[task_id];
          try {
            const ArrayNode* arr = json_obj.as<ArrayNode>();
            ICHECK_EQ(arr->size(), 3);
            workload_idxs[task_id] = Downcast<Integer>(arr->at(0)).IntValue();
            targets[task_id] = Target(Downcast<Map<String, ObjectRef>>(arr->at(1)));
            records[task_id] = TuningRecord::FromJSON(arr->at(2));
          } catch (std::runtime_error& e) {
            LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
                       << "\nThe error is: " << e.what();
          }
        });

    for (int i = 0; i < size; i++) {
      std::string key = get_database_key(workload_idxs[i], targets[i]);
      n->tuning_records_[key].insert(records[i]);
    }
  }

  // Load `n->measuremet_log` from `path_measurement_record`
  {
    std::vector<ObjectRef> json_objs =
        meta_schedule::JSONFileReadLines(path_measurement_record, num_threads, allow_missing);
    std::vector<int> workload_idxs;
    std::vector<Target> targets;
    std::vector<Array<FloatImm>> measurements;
    int size = json_objs.size();
    workload_idxs.resize(size, -1);
    targets.resize(size, Target{nullptr});
    measurements.resize(size, Array<FloatImm>({}));
    support::parallel_for_dynamic(
        0, json_objs.size(), num_threads, [&](int thread_id, int task_id) {
          const ObjectRef& json_obj = json_objs[task_id];
          try {
            const ArrayNode* arr = json_obj.as<ArrayNode>();
            ICHECK_EQ(arr->size(), 3);
            workload_idxs[task_id] = Downcast<Integer>(arr->at(0)).IntValue();
            targets[task_id] = Target(Downcast<Map<String, ObjectRef>>(arr->at(1)));
            measurements[task_id] = meta_schedule::AsFloatArray(arr->at(2));
          } catch (std::runtime_error& e) {
            LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
                       << "\nThe error is: " << e.what();
          }
        });
    for (int i = 0; i < size; i++) {
      n->measurement_records_[get_database_key(workload_idxs[i], targets[i])] = measurements[i];
    }
  }

  n->path_workload = path_workload;
  n->path_tuning_record = path_tuning_record;
  n->path_measurement_record = path_measurement_record;
  return Database(n);
}

/**************** FFI ****************/
TVM_REGISTER_NODE_TYPE(TuningRecordNode);
TVM_REGISTER_GLOBAL("relax.tuning_api.TuningRecord")
    .set_body_typed([](Trace trace, Optional<Array<FloatImm>> run_secs) {
      return TuningRecord(trace, run_secs);
    });
TVM_REGISTER_GLOBAL("relax.tuning_api.TuningRecordAsJSON")
    .set_body_method<TuningRecord>(&TuningRecordNode::AsJSON);
TVM_REGISTER_GLOBAL("relax.tuning_api.TuningRecordFromJSON").set_body_typed(TuningRecord::FromJSON);

TVM_REGISTER_OBJECT_TYPE(DatabaseNode);
TVM_REGISTER_GLOBAL("relax.tuning_api.DatabaseHasWorkload")
    .set_body_method<Database>(&DatabaseNode::HasWorkload);
TVM_REGISTER_GLOBAL("relax.tuning_api.DatabaseHasMeasurementRecord")
    .set_body_method<Database>(&DatabaseNode::HasMeasurementRecord);
TVM_REGISTER_GLOBAL("relax.tuning_api.DatabaseHasTuningRecord")
    .set_body_method<Database>(&DatabaseNode::HasTuningRecord);
TVM_REGISTER_GLOBAL("relax.tuning_api.DatabaseCommitMeasurementRecord")
    .set_body_method<Database>(&DatabaseNode::CommitMeasurementRecord);
TVM_REGISTER_GLOBAL("relax.tuning_api.DatabaseCommitWorkload")
    .set_body_method<Database>(&DatabaseNode::CommitWorkload);
TVM_REGISTER_GLOBAL("relax.tuning_api.DatabaseCommitTuningRecord")
    .set_body_method<Database>(&DatabaseNode::CommitTuningRecord);
TVM_REGISTER_GLOBAL("relax.tuning_api.DatabaseGetTopK")
    .set_body_method<Database>(&DatabaseNode::GetTopK);
TVM_REGISTER_GLOBAL("relax.tuning_api.DatabaseGetMeasurementRecord")
    .set_body_method<Database>(&DatabaseNode::GetMeasurementRecord);

TVM_REGISTER_NODE_TYPE(JSONDatabaseNode);
TVM_REGISTER_GLOBAL("relax.tuning_api.DatabaseJSONDatabase").set_body_typed(Database::JSONDatabase);
}  // namespace relax
}  // namespace tvm
