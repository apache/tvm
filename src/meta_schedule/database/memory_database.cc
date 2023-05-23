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
#include "../module_equality.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

class MemoryDatabaseNode : public DatabaseNode {
 public:
  explicit MemoryDatabaseNode(String mod_eq_name = "structural") : DatabaseNode(mod_eq_name) {}

  Array<TuningRecord> records;
  Array<Workload> workloads;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("records", &records);
    v->Visit("workloads", &workloads);
  }

  static constexpr const char* _type_key = "meta_schedule.MemoryDatabase";
  TVM_DECLARE_FINAL_OBJECT_INFO(MemoryDatabaseNode, DatabaseNode);

 public:
  bool HasWorkload(const IRModule& mod) final {
    for (const auto& workload : workloads) {
      if (GetModuleEquality().Equal(workload->mod, mod)) {
        return true;
      }
    }
    return false;
  }

  Workload CommitWorkload(const IRModule& mod) final {
    for (const auto& workload : workloads) {
      if (GetModuleEquality().Equal(workload->mod, mod)) {
        return workload;
      }
    }
    Workload workload(mod, GetModuleEquality().Hash(mod));
    workloads.push_back(workload);
    return workload;
  }

  void CommitTuningRecord(const TuningRecord& record) final { records.push_back(record); }

  Array<TuningRecord> GetTopK(const Workload& workload, int top_k) final {
    CHECK_GE(top_k, 0) << "ValueError: top_k must be non-negative";
    if (top_k == 0) {
      return {};
    }
    std::vector<TuningRecord> results;
    results.reserve(records.size());
    for (const TuningRecord& record : records) {
      if (!record->IsValid()) {
        continue;
      }
      if (record->workload.same_as(workload) ||
          WorkloadEqual(GetModuleEquality())(record->workload, workload)) {
        results.emplace_back(record);
      }
    }
    std::stable_sort(results.begin(), results.end(), SortTuningRecordByMeanRunSecs());
    if (results.size() > static_cast<size_t>(top_k)) {
      return {results.begin(), results.begin() + top_k};
    } else {
      return results;
    }
  }

  Array<TuningRecord> GetAllTuningRecords() final { return records; }

  int64_t Size() final { return records.size(); }
};

Database Database::MemoryDatabase(String mod_eq_name) {
  ObjectPtr<MemoryDatabaseNode> n = make_object<MemoryDatabaseNode>(mod_eq_name);
  n->records.clear();
  n->workloads.clear();
  return Database(n);
}

TVM_REGISTER_NODE_TYPE(MemoryDatabaseNode);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseMemoryDatabase")
    .set_body_typed(Database::MemoryDatabase);

}  // namespace meta_schedule
}  // namespace tvm
