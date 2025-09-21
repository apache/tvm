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

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

class UnionDatabaseNode : public DatabaseNode {
 public:
  ffi::Array<Database> databases;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<UnionDatabaseNode>().def_ro("databases", &UnionDatabaseNode::databases);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("meta_schedule.UnionDatabase", UnionDatabaseNode, DatabaseNode);

 public:
  ffi::Optional<TuningRecord> QueryTuningRecord(const IRModule& mod, const Target& target,
                                                const ffi::String& task_name) final {
    std::vector<TuningRecord> results;
    results.reserve(databases.size());
    for (const Database& db : databases) {
      if (ffi::Optional<TuningRecord> record = db->QueryTuningRecord(mod, target, task_name)) {
        results.push_back(record.value());
      }
    }
    std::stable_sort(results.begin(), results.end(), SortTuningRecordByMeanRunSecs());
    return results.empty() ? ffi::Optional<TuningRecord>(std::nullopt) : results[0];
  }

  bool HasWorkload(const IRModule& mod) final {
    LOG(FATAL) << "NotImplementedError: UnionDatabase.HasWorkload";
    throw;
  }

  Workload CommitWorkload(const IRModule& mod) final {
    LOG(FATAL) << "NotImplementedError: UnionDatabase.CommitWorkload";
    throw;
  }

  void CommitTuningRecord(const TuningRecord& record) final {
    LOG(FATAL) << "NotImplementedError: UnionDatabase.CommitTuningRecord";
    throw;
  }

  ffi::Array<TuningRecord> GetTopK(const Workload& workload, int top_k) final {
    LOG(FATAL) << "NotImplementedError: UnionDatabase.GetTopK";
    throw;
  }

  ffi::Array<TuningRecord> GetAllTuningRecords() final {
    LOG(FATAL) << "NotImplementedError: UnionDatabase.GetAllTuningRecords";
    throw;
  }

  int64_t Size() final {
    LOG(FATAL) << "NotImplementedError: UnionDatabase.size";
    throw;
  }
};

Database Database::UnionDatabase(ffi::Array<Database> databases) {
  ObjectPtr<UnionDatabaseNode> n = ffi::make_object<UnionDatabaseNode>();
  n->databases = std::move(databases);
  return Database(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("meta_schedule.DatabaseUnionDatabase", Database::UnionDatabase);
}

TVM_FFI_STATIC_INIT_BLOCK() { UnionDatabaseNode::RegisterReflection(); }

}  // namespace meta_schedule
}  // namespace tvm
