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

class OrderedUnionDatabaseNode : public DatabaseNode {
 public:
  ffi::Array<Database> databases;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OrderedUnionDatabaseNode>().def_ro("databases",
                                                       &OrderedUnionDatabaseNode::databases);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("meta_schedule.OrderedUnionDatabase", OrderedUnionDatabaseNode,
                                    DatabaseNode);

 public:
  ffi::Optional<TuningRecord> QueryTuningRecord(const IRModule& mod, const Target& target,
                                                const ffi::String& task_name) final {
    for (const Database& db : databases) {
      if (ffi::Optional<TuningRecord> record = db->QueryTuningRecord(mod, target, task_name)) {
        return record;
      }
    }
    return std::nullopt;
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

  ffi::Array<TuningRecord> GetTopK(const Workload& workload, int top_k) final {
    LOG(FATAL) << "NotImplementedError: OrderedUnionDatabase.GetTopK";
    throw;
  }

  ffi::Array<TuningRecord> GetAllTuningRecords() final {
    LOG(FATAL) << "NotImplementedError: OrderedUnionDatabase.GetAllTuningRecords";
    throw;
  }

  int64_t Size() final {
    LOG(FATAL) << "NotImplementedError: OrderedUnionDatabase.size";
    throw;
  }
};

Database Database::OrderedUnionDatabase(ffi::Array<Database> databases) {
  ObjectPtr<OrderedUnionDatabaseNode> n = ffi::make_object<OrderedUnionDatabaseNode>();
  n->databases = std::move(databases);
  return Database(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("meta_schedule.DatabaseOrderedUnionDatabase",
                        Database::OrderedUnionDatabase);
}

TVM_FFI_STATIC_INIT_BLOCK() { OrderedUnionDatabaseNode::RegisterReflection(); }

}  // namespace meta_schedule
}  // namespace tvm
