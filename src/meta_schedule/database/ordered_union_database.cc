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

class OrderedUnionDatabaseNode : public DatabaseNode {
 public:
  Array<Database> databases;

  void VisitAttrs(AttrVisitor* v) { v->Visit("databases", &databases); }

  static constexpr const char* _type_key = "meta_schedule.OrderedUnionDatabase";
  TVM_DECLARE_FINAL_OBJECT_INFO(OrderedUnionDatabaseNode, DatabaseNode);

 public:
  Optional<TuningRecord> QueryTuningRecord(const IRModule& mod, const Target& target,
                                           const String& task_name) final {
    for (const Database& db : databases) {
      if (Optional<TuningRecord> record = db->QueryTuningRecord(mod, target, task_name)) {
        return record;
      }
    }
    return NullOpt;
  }

  bool HasWorkload(const IRModule& mod) final {
    LOG(FATAL) << "NotImplementedError: OrderedUnionDatabase.HasWorkload";
    throw;
  }

  Workload CommitWorkload(const IRModule& mod) final {
    LOG(FATAL) << "NotImplementedError: OrderedUnionDatabase.CommitWorkload";
    throw;
  }

  void CommitTuningRecord(const TuningRecord& record) final {
    LOG(FATAL) << "NotImplementedError: OrderedUnionDatabase.CommitTuningRecord";
    throw;
  }

  Array<TuningRecord> GetTopK(const Workload& workload, int top_k) final {
    LOG(FATAL) << "NotImplementedError: OrderedUnionDatabase.GetTopK";
    throw;
  }

  Array<TuningRecord> GetAllTuningRecords() final {
    LOG(FATAL) << "NotImplementedError: OrderedUnionDatabase.GetAllTuningRecords";
    throw;
  }

  int64_t Size() final {
    LOG(FATAL) << "NotImplementedError: OrderedUnionDatabase.size";
    throw;
  }
};

Database Database::OrderedUnionDatabase(Array<Database> databases) {
  ObjectPtr<OrderedUnionDatabaseNode> n = make_object<OrderedUnionDatabaseNode>();
  n->databases = std::move(databases);
  return Database(n);
}

TVM_REGISTER_NODE_TYPE(OrderedUnionDatabaseNode);
TVM_REGISTER_GLOBAL("meta_schedule.DatabaseOrderedUnionDatabase")
    .set_body_typed(Database::OrderedUnionDatabase);

}  // namespace meta_schedule
}  // namespace tvm
