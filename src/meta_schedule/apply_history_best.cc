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
#include <tvm/te/tensor.h>

#include "./utils.h"

namespace tvm {
namespace meta_schedule {

/**************** Utility functions ****************/

template <class FunctionType, class RetType, class Callback>
Optional<RetType> GetOnlyOneFunctionCommon(const IRModule& mod, Callback on_found) {
  if (mod->functions.size() != 1) {
    return NullOpt;
  }
  for (const auto& kv : mod->functions) {
    const BaseFunc& func = kv.second;
    if (!func->IsInstance<typename FunctionType::ContainerType>()) {
      return NullOpt;
    } else {
      return on_found(kv);
    }
  }
  return NullOpt;
}

template <class FunctionType>
Optional<GlobalVar> GetOnlyOneFunctionKey(const IRModule& mod) {
  return GetOnlyOneFunctionCommon<FunctionType, GlobalVar>(mod, [](auto kv) { return kv.first; });
}

template <class FunctionType>
Optional<FunctionType> GetOnlyOneFunction(const IRModule& mod) {
  return GetOnlyOneFunctionCommon<FunctionType, FunctionType>(
      mod, [](auto kv) { return Downcast<FunctionType>(kv.second); });
}

template <class FunctionType>
bool HasOnlyOneFunction(const IRModule& mod) {
  return GetOnlyOneFunction<FunctionType>(mod).defined();
}

/**************** Context Manager ****************/

class ApplyHistoryBestInternal {
 public:
  static void EnterScope(ApplyHistoryBest ctx) { ctx.EnterWithScope(); }
  static void ExitScope(ApplyHistoryBest ctx) { ctx.ExitWithScope(); }
};

struct ApplyHistoryBestThreadLocalEntry {
  Optional<ApplyHistoryBest> ctx;
};

using ApplyHistoryBestThreadLocalStore = dmlc::ThreadLocalStore<ApplyHistoryBestThreadLocalEntry>;

Optional<ApplyHistoryBest> ApplyHistoryBest::Current() {
  return ApplyHistoryBestThreadLocalStore::Get()->ctx;
}

void ApplyHistoryBest::EnterWithScope() {
  Optional<ApplyHistoryBest>& ctx = ApplyHistoryBestThreadLocalStore::Get()->ctx;
  CHECK(!ctx.defined()) << "ValueError: Nested ApplyHistoryBest context managers are not allowed";
  ctx = *this;
}

void ApplyHistoryBest::ExitWithScope() {
  Optional<ApplyHistoryBest>& ctx = ApplyHistoryBestThreadLocalStore::Get()->ctx;
  ICHECK(ctx.defined());
  ctx = NullOpt;
}

/**************** ApplyHistoryBest ****************/

ApplyHistoryBest::ApplyHistoryBest(Database database,
                                   ApplyHistoryBestNode::FTEFilterFunc te_filter_func,
                                   PackedFunc logging_func) {
  ObjectPtr<ApplyHistoryBestNode> n = make_object<ApplyHistoryBestNode>();
  n->database = database;
  n->te_filter_func = te_filter_func;
  n->logging_func = logging_func;
  if (te_filter_func == nullptr) {
    n->te_filter_func = DefaultTaskFilter;
  }
  data_ = n;
}

Optional<IRModule> ApplyHistoryBestNode::Query(runtime::String task_name, IRModule mod,
                                               Target target, Optional<Array<IRModule>> dispatched,
                                               FTakeTuningRecord f_take_tuning_record,
                                               FDirectDispatch f_direct_dispatch) {
  ICHECK(dispatched.defined());
  ICHECK_EQ(dispatched.value().size(), 1);
  ICHECK(HasOnlyOneFunction<relay::Function>(mod)) << mod;
  IRModule prim_mod = dispatched.value()[0];
  ICHECK(HasOnlyOneFunction<tir::PrimFunc>(prim_mod)) << prim_mod;

  // Keep the original func name to be returned later.
  GlobalVar gv = GetOnlyOneFunctionKey<tir::PrimFunc>(prim_mod).value();

  // Unify func name to make sure it can be found in database
  const auto* parse_mod_func = runtime::Registry::Get("tvm.meta_schedule.tune.parse_mod");
  ICHECK(parse_mod_func) << "Parse mod function not defined!";
  prim_mod = (*parse_mod_func)(prim_mod);

  if (f_direct_dispatch != nullptr) {
    Optional<IRModule> mod = f_direct_dispatch(prim_mod);
    if (mod.defined()) {
      TVM_PY_LOG(INFO, logging_func) << "Direct dispatch applied for workload: " << task_name;
      return mod.value();
    }
  }
  if (database->HasWorkload(prim_mod)) {
    Array<TuningRecord> records = database->GetTopK(database->CommitWorkload(prim_mod), 1);
    if (records.size() == 1) {
      if (f_take_tuning_record != nullptr) {
        f_take_tuning_record(records[0]);
      }
      tir::Schedule sch =
          tir::Schedule::Traced(records[0]->workload->mod, /*seed=*/-1, /*debug_mask=*/0,
                                /*error_render_level=*/tir::ScheduleErrorRenderLevel::kNone);
      records[0]->trace->ApplyToSchedule(sch, false);
      tir::PrimFunc func = GetOnlyOneFunction<tir::PrimFunc>(sch->mod()).value();
      // Make sure we return the updated PrimFunc paired with the original func name.
      return IRModule({{gv, func}});
    }
  }
  TVM_PY_LOG(WARNING, logging_func) << "Cannot find workload: " << task_name;
  return NullOpt;
}

TVM_REGISTER_NODE_TYPE(ApplyHistoryBestNode);
TVM_REGISTER_GLOBAL("meta_schedule.ApplyHistoryBest")
    .set_body_typed([](Database database, ApplyHistoryBestNode::FTEFilterFunc te_filter_func,
                       PackedFunc logging_func) -> ApplyHistoryBest {
      return ApplyHistoryBest(database, te_filter_func, logging_func);
    });
TVM_REGISTER_GLOBAL("meta_schedule.ApplyHistoryBestEnterScope")
    .set_body_typed(ApplyHistoryBestInternal::EnterScope);
TVM_REGISTER_GLOBAL("meta_schedule.ApplyHistoryBestExitScope")
    .set_body_typed(ApplyHistoryBestInternal::ExitScope);
TVM_REGISTER_GLOBAL("meta_schedule.ApplyHistoryBestCurrent")
    .set_body_typed(ApplyHistoryBest::Current);
TVM_REGISTER_GLOBAL("meta_schedule.ApplyHistoryBestQuery")
    .set_body_method<ApplyHistoryBest>(&ApplyHistoryBestNode::Query);

}  // namespace meta_schedule
}  // namespace tvm
