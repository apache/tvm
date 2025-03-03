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
#include <set>
#include <thread>
#include <unordered_map>

#include "../module_equality.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Read lines from a json file.
 * \param path The path to the json file.
 * \param num_lines The number of threads used to concurrently parse the lines.
 * \param allow_missing Whether to create new file when the given path is not found.
 * \return An array containing lines read from the json file.
 */
std::vector<ObjectRef> JSONFileReadLines(const String& path, int num_threads, bool allow_missing) {
  std::ifstream is(path);
  if (is.good()) {
    std::vector<String> json_strs;
    for (std::string str; std::getline(is, str);) {
      json_strs.push_back(str);
    }
    int n = json_strs.size();
    std::vector<ObjectRef> json_objs;
    json_objs.resize(n);
    support::parallel_for_dynamic(0, n, num_threads, [&](int thread_id, int task_id) {
      json_objs[task_id] = JSONLoads(json_strs[task_id]);
    });
    return json_objs;
  }
  CHECK(allow_missing) << "ValueError: File doesn't exist: " << path;
  std::ofstream os(path);
  CHECK(os.good()) << "ValueError: Cannot create new file: " << path;
  return {};
}

/*!
 * \brief Append a line to a json file.
 * \param path The path to the json file.
 * \param line The line to append.
 */
void JSONFileAppendLine(const String& path, const std::string& line) {
  std::ofstream os(path, std::ofstream::app);
  CHECK(os.good()) << "ValueError: Cannot open the file to write: " << path;
  os << line << std::endl;
}

/*! \brief The default database implementation, which mimics two database tables with two files. */
class JSONDatabaseNode : public DatabaseNode {
 public:
  explicit JSONDatabaseNode(String mod_eq_name = "structural")
      : DatabaseNode(mod_eq_name),
        workloads2idx_(/*bucket_count*/ 0, WorkloadHash(), WorkloadEqual(GetModuleEquality())) {}

  /*! \brief The path to the workload table */
  String path_workload;
  /*! \brief The path to the tuning record table */
  String path_tuning_record;
  /*! \brief All the workloads in the database */
  std::unordered_map<Workload, int, WorkloadHash, WorkloadEqual> workloads2idx_;
  /*! \brief All the tuning records in the database */
  std::multiset<TuningRecord, SortTuningRecordByMeanRunSecs> tuning_records_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("path_workload", &path_workload);
    v->Visit("path_tuning_record", &path_tuning_record);
    // `workloads2idx_` is not visited
    // `tuning_records_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.JSONDatabase";
  TVM_DECLARE_FINAL_OBJECT_INFO(JSONDatabaseNode, DatabaseNode);

 public:
  bool HasWorkload(const IRModule& mod) {
    return workloads2idx_.find(Workload(mod, GetModuleEquality().Hash(mod))) !=
           workloads2idx_.end();
  }

  Workload CommitWorkload(const IRModule& mod) {
    // Try to insert `mod` into `workloads_`
    auto [it, inserted] =
        this->workloads2idx_.emplace(Workload(mod, GetModuleEquality().Hash(mod)), -1);
    Workload workload = it->first;
    // If `mod` is new in `workloads2idx_`, append it to the workload file
    if (inserted) {
      it->second = static_cast<int>(this->workloads2idx_.size()) - 1;
      JSONFileAppendLine(this->path_workload, JSONDumps(workload->AsJSON()));
    }
    return it->first;
  }

  void CommitTuningRecord(const TuningRecord& record) {
    this->tuning_records_.insert(record);
    JSONFileAppendLine(this->path_tuning_record,
                       JSONDumps(Array<ObjectRef>{
                           /*workload_index=*/Integer(this->workloads2idx_.at(record->workload)),
                           /*tuning_record=*/record->AsJSON()  //
                       }));
  }

  Array<TuningRecord> GetTopK(const Workload& workload, int top_k) {
    CHECK_GE(top_k, 0) << "ValueError: top_k must be non-negative";
    if (top_k == 0) {
      return {};
    }
    Array<TuningRecord> results;
    results.reserve(top_k);
    for (const TuningRecord& record : this->tuning_records_) {
      auto run_secs = record->run_secs;
      if (!record->IsValid()) {
        continue;
      }
      if (record->workload.same_as(workload) ||
          WorkloadEqual(GetModuleEquality())(record->workload, workload)) {
        results.push_back(record);
        if (results.size() == static_cast<size_t>(top_k)) {
          break;
        }
      }
    }
    return results;
  }

  Array<TuningRecord> GetAllTuningRecords() {
    Array<TuningRecord> results;
    results.reserve(Size());
    for (const TuningRecord& record : this->tuning_records_) {
      results.push_back(record);
    }
    return results;
  }

  int64_t Size() { return tuning_records_.size(); }
};

Database Database::JSONDatabase(String path_workload, String path_tuning_record, bool allow_missing,
                                String mod_eq_name) {
  int num_threads = std::thread::hardware_concurrency();
  ObjectPtr<JSONDatabaseNode> n = make_object<JSONDatabaseNode>(mod_eq_name);
  // Load `n->workloads2idx_` from `path_workload`
  std::vector<Workload> workloads;
  {
    std::vector<ObjectRef> json_objs = JSONFileReadLines(path_workload, num_threads, allow_missing);
    int n_objs = json_objs.size();
    n->workloads2idx_.reserve(n_objs);
    workloads.reserve(n_objs);
    for (int i = 0; i < n_objs; ++i) {
      Workload workload = Workload::FromJSON(json_objs[i]);
      auto recalc_hash = n->GetModuleEquality().Hash(workload->mod);
      // Todo(tvm-team): re-enable the shash check when we get environment
      // independent structural hash values.
      if (recalc_hash != workload->shash) {
        ObjectPtr<WorkloadNode> wkl = make_object<WorkloadNode>(*workload.get());
        wkl->shash = recalc_hash;
        workload = Workload(wkl);
      }
      n->workloads2idx_.emplace(workload, i);
      workloads.push_back(workload);
    }
  }
  // Load `n->tuning_records_` from `path_tuning_record`
  {
    std::vector<ObjectRef> json_objs =
        JSONFileReadLines(path_tuning_record, num_threads, allow_missing);
    std::vector<TuningRecord> records;
    records.resize(json_objs.size(), TuningRecord{nullptr});
    support::parallel_for_dynamic(
        0, json_objs.size(), num_threads, [&](int thread_id, int task_id) {
          const ObjectRef& json_obj = json_objs[task_id];
          Workload workload{nullptr};
          try {
            const ArrayNode* arr = json_obj.as<ArrayNode>();
            ICHECK_EQ(arr->size(), 2);
            int64_t workload_index = Downcast<runtime::Int>(arr->at(0));
            ICHECK(workload_index >= 0 && static_cast<size_t>(workload_index) < workloads.size());
            workload = workloads[workload_index];
            records[task_id] = TuningRecord::FromJSON(arr->at(1), workload);
          } catch (std::runtime_error& e) {
            LOG(FATAL) << "ValueError: Unable to parse TuningRecord, on line " << (task_id + 1)
                       << " of file " << path_tuning_record << ". The workload is:\n"
                       << (workload.defined() ? workload->mod->Script() : "(null)")
                       << "\nThe JSONObject of TuningRecord is:\n"
                       << json_obj << "\nThe error message is:\n"
                       << e.what();
          }
        });
    for (const TuningRecord& record : records) {
      n->tuning_records_.insert(record);
    }
  }
  n->path_workload = path_workload;
  n->path_tuning_record = path_tuning_record;
  return Database(n);
}

TVM_REGISTER_NODE_TYPE(JSONDatabaseNode);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseJSONDatabase").set_body_typed(Database::JSONDatabase);

}  // namespace meta_schedule
}  // namespace tvm
