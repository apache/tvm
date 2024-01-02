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
 * \file tvm/relax/transform/meta_schedule.cc
 * \brief Pass for meta_schedule tuning
 */
#include <tvm/meta_schedule/database.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/tuning_api.h>
#include <tvm/tir/transform.h>

#include "../src/meta_schedule/module_equality.h"
#include "../src/meta_schedule/trace_apply.h"

namespace tvm {
namespace relax {
namespace transform {

class MetaScheduleTuner {
 public:
  explicit MetaScheduleTuner(Target target, String work_dir, Integer max_trials_global,
                             Integer max_trials_per_task, Optional<Array<String>> op_names,
                             Map<String, runtime::NDArray> params = {})
      : target_(target),
        work_dir_(work_dir),
        max_trials_global_(max_trials_global),
        max_trials_per_task_(max_trials_per_task),
        op_names_(op_names),
        params_(params) {
    // candgen_func_ = runtime::Registry::Get("relax.tuning_api.default_generate_candidate");
    // ICHECK(candgen_func_) << "Default candidate generation function is not found.";
    normalize_mod_func_ = runtime::Registry::Get("tvm.meta_schedule.normalize_mod");
    ICHECK(normalize_mod_func_) << "Normalization function is not found.";
  }

  // TODO(@sunggg): Currently, only supports basic arguments.
  IRModule TuneIRMod(IRModule mod, transform::PassContext ctx) {
    Choice choice(
        "tvm.meta_schedule.tune_relax",
        {params_, target_, work_dir_, max_trials_global_, max_trials_per_task_, op_names_},
        "relax.tuning_api.Choice.default_constr_func", {});
    Knob knob("meta_schedule.tune_irmod", {{"0", choice}});
    knob->Apply(mod, "0");
    /*
    // TODO(@sunggg): revisit when we have a solution for large params
    Trace trace = Downcast<Trace>(ctx->GetCurrentTrace());
    ctx->PopTrace();
    Array<Trace> candidates = (*candgen_func_)(Array<Knob>({knob}), trace);
    ICHECK(candidates.size() == 1);
    Trace best_trace = candidates[0];
    ctx->PushTrace(best_trace);
    */
    // since we separate tuning from application, return original IRModule
    return mod;
  }

  // TODO(@sunggg): Currently, only supports basic arguments.
  tir::PrimFunc TuneTIR(tir::PrimFunc f, transform::PassContext ctx) {
    // TODO(@sunggg): Whenever we tune tir, assume we start a new trace w/o pushing to the trace
    // stack. Revisit later when we collect more usecases.
    Choice choice("tvm.meta_schedule.tune_tir", {target_, work_dir_, max_trials_global_},
                  "relax.tuning_api.Choice.default_constr_func", {});
    Knob knob("meta_schedule.tune_primfunc", {{"0", choice}});
    knob->Apply((*normalize_mod_func_)(f), "0");
    /*
    // TODO(@sunggg): revisit when we have a solution for large params
    Trace trace = Trace((*normalize_mod_func_)(f), {}, {});
    Array<Trace> candidates = (*candgen_func_)(Array<Knob>({knob}), trace);
    ICHECK(candidates.size() == 1);
    */
    // since we separate tuning from application, return original IRModule
    return f;
  }

 private:
  Target target_;
  String work_dir_;
  Integer max_trials_global_;
  Integer max_trials_per_task_;
  Optional<Array<String>> op_names_;
  Map<String, runtime::NDArray> params_;
  // const runtime::PackedFunc* candgen_func_;
  const runtime::PackedFunc* normalize_mod_func_;
};

Pass MetaScheduleApplyDatabase(Optional<String> work_dir, bool enable_warning = false) {
  using tvm::meta_schedule::Database;
  Target target = Target::Current(false);
  const runtime::PackedFunc* normalize_mod_func_ =
      runtime::Registry::Get("tvm.meta_schedule.normalize_mod");
  ICHECK(normalize_mod_func_) << "Normalization function is not found.";

  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext ctx) {
    Database database{nullptr};
    if (Database::Current().defined()) {
      database = Database::Current().value();
    } else {
      ICHECK(work_dir.defined());
      String path_workload = work_dir.value() + "/database_workload.json";
      String path_tuning_record = work_dir.value() + "/database_tuning_record.json";
      LOG(WARNING) << "Creating JSONDatabase. Workload at: " << path_workload
                   << ", Tuning records at: " << path_tuning_record;
      database = meta_schedule::Database::JSONDatabase(path_workload, path_tuning_record, true);
    }

    Map<GlobalVar, BaseFunc> result;
    auto mod_eq_structural = meta_schedule::ModuleEquality::Create("ignore-ndarray");
    for (const auto& iter : mod->functions) {
      GlobalVar gv = iter.first;
      BaseFunc base_func = iter.second;
      if (const auto* prim_func_node = base_func.as<tir::PrimFuncNode>()) {
        tir::PrimFunc prim_func = GetRef<tir::PrimFunc>(prim_func_node);

        IRModule tir_mod = (*normalize_mod_func_)(prim_func);
        if (Optional<meta_schedule::TuningRecord> opt_record =
                database->QueryTuningRecord(tir_mod, target, gv->name_hint)) {
          meta_schedule::TuningRecord record = opt_record.value();
          tir::Schedule sch{nullptr};
          if (!mod_eq_structural->Equal(tir_mod, record->workload->mod)) {
            // When the database lookup succeeds while structural equality check fails,
            // it implies that the anchor block based equality has been used during tuning.
            // The trace in the record cannot directly be applied to this query module.
            sch = tir::Schedule::Traced(
                tir_mod, /*seed=*/-1, /*debug_mask=*/0,
                /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);
            meta_schedule::ScheduleUsingAnchorTrace(sch, record->trace, target);
          } else {
            sch = tir::Schedule::Traced(
                record->workload->mod, /*seed=*/-1, /*debug_mask=*/0,
                /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);
            record->trace->ApplyToSchedule(sch, /*remove_postproc=*/false);
          }
          IRModule new_mod = sch->mod();
          ICHECK_EQ(new_mod->functions.size(), 1);
          BaseFunc new_base_func = (*new_mod->functions.begin()).second;
          ICHECK(new_base_func->IsInstance<tir::PrimFuncNode>());
          tir::PrimFunc tuned_prim_func = Downcast<tir::PrimFunc>(new_base_func);
          // maintain the original attributes
          tir::PrimFunc new_prim_func = tir::PrimFunc(/*params=*/tuned_prim_func->params,
                                                      /*body=*/tuned_prim_func->body,
                                                      /*ret_type=*/tuned_prim_func->ret_type,
                                                      /*buffer_map=*/tuned_prim_func->buffer_map,
                                                      /*attrs=*/prim_func->attrs);
          new_prim_func = WithAttr(std::move(new_prim_func), tir::attr::kIsScheduled, Bool(true));
          result.Set(gv, new_prim_func);
          continue;
        } else if (enable_warning) {
          LOG(WARNING) << "Tuning record is not found for primfunc: " << gv->name_hint;
        }
      }
      result.Set(gv, base_func);
    }
    return IRModule(result,       // functions
                    {},           // type_definitions
                    {},           // import_set
                    {},           // map
                    mod->attrs);  // attrs);
  };
  return CreateModulePass(pass_func, 0, "MetaScheduleApplyDatabase", {});
}

Pass MetaScheduleTuneIRMod(Map<String, runtime::NDArray> params, String work_dir,
                           Integer max_trials_global,
                           Optional<Integer> max_trials_per_task = NullOpt,
                           Optional<Array<String>> op_names = NullOpt) {
  Target target = Target::Current(false);
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext ctx) {
    auto max_trials_task = max_trials_per_task.value_or(max_trials_global);
    return MetaScheduleTuner(target, work_dir, max_trials_global, max_trials_task, op_names, params)
        .TuneIRMod(m, ctx);
  };
  return CreateModulePass(/*pass function*/ pass_func, /*opt level*/ 0,
                          /*pass name*/ "MetaScheduleTuneIRModule",
                          /*required*/ {},
                          /*traceable*/ true);
}

Pass MetaScheduleTuneTIR(String work_dir, Integer max_trials_global) {
  Target target = Target::Current(false);
  runtime::TypedPackedFunc<tir::PrimFunc(tir::PrimFunc, IRModule, PassContext)> pass_func =
      [=](tir::PrimFunc f, IRModule mod, PassContext ctx) {
        return MetaScheduleTuner(target, work_dir, max_trials_global, max_trials_global, NullOpt)
            .TuneTIR(f, ctx);
      };
  return tir::transform::CreatePrimFuncPass(/*pass function*/ pass_func, /*opt level*/ 0,
                                            /*pass name*/ "MetaScheduleTuneTIR",
                                            /*required*/ {},
                                            /*traceable*/ true);
}

TVM_REGISTER_GLOBAL("relax.transform.MetaScheduleApplyDatabase")
    .set_body_typed(MetaScheduleApplyDatabase);
TVM_REGISTER_GLOBAL("relax.transform.MetaScheduleTuneIRMod").set_body_typed(MetaScheduleTuneIRMod);
TVM_REGISTER_GLOBAL("relax.transform.MetaScheduleTuneTIR").set_body_typed(MetaScheduleTuneTIR);
}  // namespace transform
}  // namespace relax
}  // namespace tvm
