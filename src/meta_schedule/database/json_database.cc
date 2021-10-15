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
#include <unordered_map>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

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
    double a_time = Mean(a->run_secs);
    double b_time = Mean(b->run_secs);
    return a_time < b_time;
  }
};

/*! \brief The default database implementation, which mimics two database tables with two files. */
class JSONDatabaseNode : public DatabaseNode {
 public:
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
  Workload CommitWorkload(const IRModule& mod) {
    // Try to insert `mod` into `workloads_`
    decltype(this->workloads2idx_)::iterator it;
    bool inserted = false;
    std::tie(it, inserted) =
        this->workloads2idx_.emplace(Workload(mod, tvm::StructuralHash()(mod)), -1);
    Workload workload = it->first;
    // If `mod` is new in `workloads2idx_`, append it to the workload file
    if (inserted) {
      it->second = static_cast<int>(this->workloads2idx_.size()) - 1;
      JSONFileAppendLine(this->path_workload, JSONObj2Str(workload->AsJSON()));
    }
    return it->first;
  }

  void CommitTuningRecord(const TuningRecord& record) {
    this->tuning_records_.insert(record);
    JSONFileAppendLine(this->path_tuning_record,
                       JSONObj2Str(Array<ObjectRef>{
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
    int counter = 0;
    for (const TuningRecord& record : this->tuning_records_) {
      if (WorkloadEqual()(record->workload, workload)) {
        results.push_back(record);
        if (++counter == top_k) {
          break;
        }
      }
    }
    return results;
  }

  int64_t Size() { return tuning_records_.size(); }
};

Database Database::JSONDatabase(String path_workload, String path_tuning_record,
                                bool allow_missing) {
  ObjectPtr<JSONDatabaseNode> n = make_object<JSONDatabaseNode>();
  // Load `n->workloads2idx_` from `path_workload`
  std::vector<Workload> workloads;
  {
    Array<ObjectRef> json_objs = JSONStr2Obj(JSONFileReadLines(path_workload, allow_missing));
    int n_objs = json_objs.size();
    n->workloads2idx_.reserve(n_objs);
    workloads.reserve(n_objs);
    for (int i = 0; i < n_objs; ++i) {
      Workload workload = Workload::FromJSON(json_objs[i]);
      n->workloads2idx_.emplace(workload, i);
      workloads.push_back(workload);
    }
  }
  // Load `n->tuning_records_` from `path_tuning_record`
  {
    Array<ObjectRef> json_objs = JSONStr2Obj(JSONFileReadLines(path_tuning_record, allow_missing));
    for (const ObjectRef& json_obj : json_objs) {
      int workload_index = -1;
      ObjectRef tuning_record{nullptr};
      try {
        const ArrayNode* arr = json_obj.as<ArrayNode>();
        ICHECK_EQ(arr->size(), 2);
        workload_index = Downcast<Integer>(arr->at(0));
        tuning_record = arr->at(1);
      } catch (std::runtime_error& e) {
        LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
                   << "\nThe error is: " << e.what();
      }
      n->tuning_records_.insert(TuningRecord::FromJSON(tuning_record, workloads[workload_index]));
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
