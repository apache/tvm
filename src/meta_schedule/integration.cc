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
#include <tvm/meta_schedule/integration.h>
#include <tvm/relay/function.h>
#include <tvm/tir/function.h>

#include "./utils.h"

namespace tvm {
namespace meta_schedule {

/**************** Utility functions ****************/

template <class FunctionType>
bool HasOnlyOneFunction(const IRModule& mod) {
  if (mod->functions.size() != 1) {
    return false;
  }
  for (const auto& kv : mod->functions) {
    const BaseFunc& func = kv.second;
    if (!func->IsInstance<typename FunctionType::ContainerType>()) {
      return false;
    }
  }
  return true;
}

/**************** ExtractedTask ****************/

ExtractedTask::ExtractedTask(String task_name, IRModule mod, Target target,
                             Array<IRModule> dispatched) {
  ObjectPtr<ExtractedTaskNode> n = make_object<ExtractedTaskNode>();
  n->task_name = task_name;
  n->mod = mod;
  n->target = target;
  n->dispatched = dispatched;
  data_ = n;
}

/**************** MetaScheduleContext ****************/

struct MetaScheduleContextThreadLocalEntry {
  Optional<MetaScheduleContext> ctx;
};

using MetaScheduleContextThreadLocalStore =
    dmlc::ThreadLocalStore<MetaScheduleContextThreadLocalEntry>;

Optional<MetaScheduleContext> MetaScheduleContext::Current() {
  return MetaScheduleContextThreadLocalStore::Get()->ctx;
}

void MetaScheduleContext::EnterWithScope() {
  Optional<MetaScheduleContext>& ctx = MetaScheduleContextThreadLocalStore::Get()->ctx;
  CHECK(!ctx.defined())
      << "ValueError: Nested MetaScheduleContext context managers are not allowed";
  ctx = *this;
}

void MetaScheduleContext::ExitWithScope() {
  Optional<MetaScheduleContext>& ctx = MetaScheduleContextThreadLocalStore::Get()->ctx;
  ICHECK(ctx.defined());
  ctx = NullOpt;
}

Optional<ObjectRef> MetaScheduleContext::QueryInsideWithScope(
    runtime::String task_name, IRModule mod, Target target, Optional<Array<IRModule>> dispatched) {
  if (Optional<MetaScheduleContext> ctx = MetaScheduleContext::Current()) {
    return ctx.value()->Query(task_name, mod, target, dispatched);
  }
  return NullOpt;
}

/**************** TaskExtraction ****************/

TaskExtraction::TaskExtraction() {
  ObjectPtr<TaskExtractionNode> n = make_object<TaskExtractionNode>();
  n->tasks = Array<ExtractedTask>();
  data_ = n;
}

Optional<ObjectRef> TaskExtractionNode::Query(runtime::String task_name, IRModule mod,
                                              Target target, Optional<Array<IRModule>> dispatched) {
  ICHECK(dispatched.defined());
  ICHECK_EQ(dispatched.value().size(), 1);
  IRModule prim_mod = dispatched.value()[0];
  ICHECK(HasOnlyOneFunction<tir::PrimFunc>(prim_mod)) << prim_mod;
  ICHECK(HasOnlyOneFunction<relay::Function>(mod)) << mod;
  tasks.push_back(ExtractedTask(task_name, mod, target, {prim_mod}));
  return NullOpt;
}

/**************** ApplyHistoryBest ****************/

ApplyHistoryBest::ApplyHistoryBest(Database database) {
  ObjectPtr<ApplyHistoryBestNode> n = make_object<ApplyHistoryBestNode>();
  n->database = database;
  data_ = n;
}

Optional<ObjectRef> ApplyHistoryBestNode::Query(runtime::String task_name, IRModule mod,
                                                Target target,
                                                Optional<Array<IRModule>> dispatched) {
  ICHECK(dispatched.defined());
  ICHECK_EQ(dispatched.value().size(), 1);
  ICHECK(HasOnlyOneFunction<relay::Function>(mod)) << mod;
  IRModule prim_mod = dispatched.value()[0];
  ICHECK(HasOnlyOneFunction<tir::PrimFunc>(prim_mod)) << prim_mod;
  // Unify func name to make sure it can be found in database
  const auto* parse_mod_func = runtime::Registry::Get("tvm.meta_schedule.tune.parse_mod");
  ICHECK(parse_mod_func) << "Parse mod function not defined!";
  prim_mod = (*parse_mod_func)(prim_mod);
  if (database->HasWorkload(prim_mod)) {
    Array<TuningRecord> records = database->GetTopK(database->CommitWorkload(prim_mod), 1);
    if (records.size() == 1) {
      LOG(INFO) << "Applied history best for " << task_name << ".";
      tir::Schedule sch =
          tir::Schedule::Traced(records[0]->workload->mod, /*seed=*/-1, /*debug_mask=*/0,
                                /*error_render_level=*/tir::ScheduleErrorRenderLevel::kNone);
      records[0]->trace->ApplyToSchedule(sch, false);
      return sch->mod();
    }
  }
  return NullOpt;
}

/**************** FFI ****************/

class MetaScheduleContextInternal {
 public:
  static void EnterScope(MetaScheduleContext ctx) { ctx.EnterWithScope(); }
  static void ExitScope(MetaScheduleContext ctx) { ctx.ExitWithScope(); }
};

TVM_REGISTER_NODE_TYPE(ExtractedTaskNode);
TVM_REGISTER_OBJECT_TYPE(MetaScheduleContextNode);
TVM_REGISTER_NODE_TYPE(TaskExtractionNode);
TVM_REGISTER_NODE_TYPE(ApplyHistoryBestNode);

TVM_REGISTER_GLOBAL("meta_schedule.ExtractedTask")
    .set_body_typed([](String task_name, IRModule mod, Target target,
                       Array<IRModule> dispatched) -> ExtractedTask {
      return ExtractedTask(task_name, mod, target, dispatched);
    });
TVM_REGISTER_GLOBAL("meta_schedule.MetaScheduleContextEnterScope")
    .set_body_typed(MetaScheduleContextInternal::EnterScope);
TVM_REGISTER_GLOBAL("meta_schedule.MetaScheduleContextExitScope")
    .set_body_typed(MetaScheduleContextInternal::ExitScope);
TVM_REGISTER_GLOBAL("meta_schedule.MetaScheduleContextCurrent")
    .set_body_typed(MetaScheduleContext::Current);
TVM_REGISTER_GLOBAL("meta_schedule.MetaScheduleContextQueryInsideWithScope")
    .set_body_typed(MetaScheduleContext::QueryInsideWithScope);
TVM_REGISTER_GLOBAL("meta_schedule.MetaScheduleContextQuery")
    .set_body_method<MetaScheduleContext>(&MetaScheduleContextNode::Query);
TVM_REGISTER_GLOBAL("meta_schedule.TaskExtraction").set_body_typed([]() -> TaskExtraction {
  return TaskExtraction();
});
TVM_REGISTER_GLOBAL("meta_schedule.ApplyHistoryBest")
    .set_body_typed([](Database database) -> ApplyHistoryBest {
      return ApplyHistoryBest(database);
    });

}  // namespace meta_schedule
}  // namespace tvm
