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

class MemoryDatabaseNode : public DatabaseNode {
 public:
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
      if (StructuralEqual()(workload->mod, mod)) {
        return true;
      }
    }
    return false;
  }

  Workload CommitWorkload(const IRModule& mod) {
    for (const auto& workload : workloads) {
      if (StructuralEqual()(workload->mod, mod)) {
        return workload;
      }
    }
    Workload workload(mod, StructuralHash()(mod));
    workloads.push_back(workload);
    return workload;
  }

  void CommitTuningRecord(const TuningRecord& record) { records.push_back(record); }

  Array<TuningRecord> GetTopK(const Workload& workload, int top_k) {
    std::vector<std::pair<double, TuningRecord>> results;
    results.reserve(this->records.size());
    for (const TuningRecord& record : records) {
      if (!record->run_secs.defined()) {
        continue;
      }
      Array<FloatImm> run_secs = record->run_secs.value();
      if (run_secs.empty()) {
        continue;
      }
      if (record->workload.same_as(workload)) {
        double sum = 0.0;
        for (const FloatImm& i : run_secs) {
          sum += i->value;
        }
        results.emplace_back(sum / run_secs.size(), record);
      }
    }
    std::sort(results.begin(), results.end());
    auto begin = results.begin();
    auto end = results.end();
    if (static_cast<int>(results.size()) > top_k) {
      end = begin + top_k;
    }
    Array<TuningRecord> ret;
    ret.reserve(end - begin);
    while (begin != end) {
      ret.push_back(begin->second);
      ++begin;
    }
    return ret;
  }

  Array<TuningRecord> GetAllTuningRecords() { return records; }

  int64_t Size() { return records.size(); }
};

Database Database::MemoryDatabase() {
  ObjectPtr<MemoryDatabaseNode> n = make_object<MemoryDatabaseNode>();
  n->records.clear();
  n->workloads.clear();
  return Database(n);
}

TVM_REGISTER_NODE_TYPE(MemoryDatabaseNode);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseMemoryDatabase")
    .set_body_typed(Database::MemoryDatabase);

}  // namespace meta_schedule
}  // namespace tvm
