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

namespace tvm {
namespace relax {
namespace transform {

class MetaScheduleTuner {
 public:
  explicit MetaScheduleTuner(Target target, String work_dir, Integer max_trials_global,
                             Map<String, runtime::NDArray> params = {})
      : target_(target),
        work_dir_(work_dir),
        max_trials_global_(max_trials_global),
        params_(params) {
    candgen_func_ = runtime::Registry::Get("relax.tuning_api.default_generate_candidate");
    ICHECK(candgen_func_) << "Default candidate generation function is not found.";
    normalize_mod_func_ = runtime::Registry::Get("tvm.meta_schedule.normalize_mod");
    ICHECK(normalize_mod_func_) << "Normalization function is not found.";
  }

  // TODO(@sunggg): Currently, only supports basic arguments.
  IRModule TuneIRMod(IRModule mod, transform::PassContext ctx) {
    Trace trace = Downcast<Trace>(ctx->GetCurrentTrace());
    ctx->PopTrace();
    Choice choice("tvm.meta_schedule.tune_relax", {params_, target_, work_dir_, max_trials_global_},
                  "relax.tuning_api.Choice.default_constr_func", {});
    Knob knob("meta_schedule.tune_irmod", {{"0", choice}});
    Array<Trace> candidates = (*candgen_func_)(Array<Knob>({knob}), trace);
    ICHECK(candidates.size() == 1);
    Trace best_trace = candidates[0];
    ctx->PushTrace(best_trace);
    // since we separate tuning from application, return original IRModule
    return mod;
  }

  // TODO(@sunggg): Currently, only supports basic arguments.
  tir::PrimFunc TuneTIR(tir::PrimFunc f, transform::PassContext ctx) {
    // TODO(@sunggg): Whenever we tune tir, assume we start a new trace w/o pushing to the trace
    // stack. Revisit later when we collect more usecases.
    Trace trace = Trace((*normalize_mod_func_)(f), {}, {});

    Choice choice("tvm.meta_schedule.tune_tir", {target_, work_dir_, max_trials_global_},
                  "relax.tuning_api.Choice.default_constr_func", {});
    Knob knob("meta_schedule.tune_primfunc", {{"0", choice}});
    Array<Trace> candidates = (*candgen_func_)(Array<Knob>({knob}), trace);
    ICHECK(candidates.size() == 1);
    // since we separate tuning from application, return original IRModule
    return f;
  }

 private:
  Target target_;
  String work_dir_;
  Integer max_trials_global_;
  Map<String, runtime::NDArray> params_;
  const runtime::PackedFunc* candgen_func_;
  const runtime::PackedFunc* normalize_mod_func_;
};

Pass MetaScheduleApplyDatabase(Optional<String> work_dir) {
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
    for (const auto& iter : mod->functions) {
      GlobalVar gv = iter.first;
      BaseFunc base_func = iter.second;
      if (const auto* prim_func_node = base_func.as<tir::PrimFuncNode>()) {
        tir::PrimFunc prim_func = GetRef<tir::PrimFunc>(prim_func_node);

        IRModule tir_mod = (*normalize_mod_func_)(prim_func);
        if (Optional<tir::Schedule> sch = database->QuerySchedule(tir_mod, target, gv->name_hint)) {
          IRModule new_mod = sch.value()->mod();
          ICHECK_EQ(new_mod->functions.size(), 1);
          BaseFunc new_base_func = (*new_mod->functions.begin()).second;
          ICHECK(new_base_func->IsInstance<tir::PrimFuncNode>());
          tir::PrimFunc new_prim_func = Downcast<tir::PrimFunc>(new_base_func);
          // copy the original attrs
          new_prim_func = WithAttrs(std::move(new_prim_func), {prim_func->attrs->dict});
          new_prim_func = WithAttr(std::move(new_prim_func), tir::attr::kIsScheduled, Bool(true));
          result.Set(gv, new_prim_func);
          continue;
        } else {
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
                           Integer max_trials_global) {
  Target target = Target::Current(false);
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext ctx) {
    return MetaScheduleTuner(target, work_dir, max_trials_global, params).TuneIRMod(m, ctx);
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
        return MetaScheduleTuner(target, work_dir, max_trials_global).TuneTIR(f, ctx);
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
