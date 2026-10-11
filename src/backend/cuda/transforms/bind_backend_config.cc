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

/*! \file bind_backend_config.cc
 *  \brief Resolve CUDA configuration before any target-sensitive lowering.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/unique_name_supply.h>
#include <tvm/target/target.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tirx {
namespace transform {

class BackendConfigBinder : public StmtExprMutator {
 public:
  BackendConfigBinder(Target target, ffi::String defaults)
      : target_(target),
        defaults_(defaults),
        resolve_(ffi::Function::GetGlobalRequired("cuda.resolve_backend_config")) {}

  Function Apply(Function func) {
    TVM_FFI_CHECK(!func->GetAttr<ffi::String>("tirx.cuda_arch").has_value(), ValueError)
        << "tirx.cuda_arch was removed; set backend_config['cuda']['arch'] on device_entry or "
           "tvm.compile";
    auto resolved =
        resolve_(func->GetAttr<ffi::String>("backend_config").value_or(""), defaults_, target_)
            .cast<ffi::Array<ffi::Any>>();
    target_ = resolved[0].cast<Target>();
    defaults_ = resolved[1].cast<ffi::String>();
    if (!func->body.has_value()) return func;
    auto body =
        Mutate(func->body.value(), InplaceMode::kDisallow).ValueOrUnchanged(func->body.value());
    func.CopyOnWrite()->body = body;
    return WithAttrs(func, {{tvm::attr::kTarget, target_}, {"backend_config", defaults_}});
  }

 private:
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final {
    if (op->op->name != "tirx.device_entry" && op->op->name != "tirx.device_scope") {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    auto local = op->attrs->dict.Get("backend_config");
    auto resolved = resolve_(local.has_value() ? local.value().cast<ffi::String>() : ffi::String(),
                             defaults_, target_)
                        .cast<ffi::Array<ffi::Any>>();
    auto attrs = op->attrs->dict;
    attrs.Set(tvm::attr::kTarget, resolved[0]);
    attrs.Set("backend_config", resolved[1]);
    auto body = Mutate(op->body, InplaceMode::kDisallow).ValueOrUnchanged(op->body);
    return RegionStmt(op->op, op->args, op->body_params, DictAttrs(attrs), body, op->result_vars,
                      op->loc);
  }
  Target target_;
  ffi::String defaults_;
  ffi::Function resolve_;
};

Pass BindBackendConfig(ffi::String defaults) {
  auto transform = [defaults](Function func, IRModule mod, PassContext ctx) -> Function {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.has_value() || target.value()->kind->name != "cuda") return func;
    return BackendConfigBinder(target.value(), defaults).Apply(func);
  };
  return CreateFunctionPass(transform, 0, "cuda.BindBackendConfig");
}

// Private device helpers belong to a compilation group. Specialize their
// transitive closure before dtype legalization so every helper sees the same
// target and options as its callers. No device symbol crosses binary boundaries.
class DeviceHelperSpecializer : public StmtExprMutator {
 public:
  DeviceHelperSpecializer(IRModule input, IRModule output, UniqueNameSupply names, Target target,
                          ffi::String config, bool active = true)
      : input_(input),
        output_(output),
        names_(names),
        target_(target),
        config_(config),
        active_(active) {}

  Function Rewrite(Function func) {
    if (!func->body.has_value()) return func;
    auto body =
        Mutate(func->body.value(), InplaceMode::kDisallow).ValueOrUnchanged(func->body.value());
    func.CopyOnWrite()->body = body;
    return func;
  }

  std::unordered_set<const GlobalVarNode*> originals;

 private:
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final {
    if (op->op->name != "tirx.device_entry") {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    auto entry_target = op->attrs->dict.Get(tvm::attr::kTarget);
    if (!entry_target || entry_target.value().cast<Target>()->kind->name != "cuda") {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    auto saved_target = target_;
    auto saved_config = config_;
    auto saved_replacements = replacements_;
    bool saved_active = active_;
    target_ = entry_target.value().cast<Target>().WithoutHost();
    config_ = op->attrs->dict.at("backend_config").cast<ffi::String>();
    replacements_ = {};
    active_ = true;
    auto result = StmtExprMutator::Mutate_(op, inplace_mode);
    target_ = saved_target;
    config_ = saved_config;
    replacements_ = saved_replacements;
    active_ = saved_active;
    return result;
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    auto result =
        StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Expr>(op));
    auto call = result.as_or_throw<Call>();
    auto callee = call->op.as<GlobalVar>();
    if (!active_ || !callee.has_value() || !input_->functions.count(callee.value())) return result;
    originals.insert(callee.value().get());
    auto original = input_->Lookup(callee.value()).as_or_throw<Function>();
    auto original_target = original->GetAttr<Target>(tvm::attr::kTarget);
    TVM_FFI_CHECK(original_target.has_value() && original_target.value()->kind->name == "cuda",
                  ValueError)
        << "CUDA device function cannot call a non-CUDA helper";
    TVM_FFI_CHECK(
        original->GetAttr<CallingConv>(tvm::attr::kCallingConv, CallingConv::kDefault).value() ==
            CallingConv::kDefault,
        ValueError)
        << "CUDA device calls must refer to private device helpers";
    GlobalVar replacement(ffi::UnsafeInit{});
    if (auto existing = replacements_.Get(callee.value())) {
      replacement = existing.value();
    } else {
      bool same_group = original_target.value()->str() == target_->str() &&
                        original->GetAttr<ffi::String>("backend_config").value_or("") == config_;
      replacement = same_group
                        ? callee.value()
                        : GlobalVar(names_->FreshName(callee.value()->name_hint + "_cuda", false));
      replacements_.Set(callee.value(), replacement);
      auto helper =
          WithAttrs(original, {{tvm::attr::kTarget, target_}, {"backend_config", config_}});
      helper = Rewrite(helper);
      output_->Add(replacement, helper);
    }
    call.CopyOnWrite()->op = replacement;
    return call;
  }
  IRModule input_, output_;
  UniqueNameSupply names_;
  Target target_;
  ffi::String config_;
  bool active_;
  ffi::Map<GlobalVar, GlobalVar> replacements_;
};

// Entry targets must also govern helper tile dispatch, which precedes splitting.
// Clone before lowering, then discard only originals made unreachable by cloning.
class HelperCallCollector : public StmtExprVisitor {
 public:
  std::vector<GlobalVar> calls;
  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) final {
    if (auto gv = op->op.as<GlobalVar>()) calls.push_back(gv.value());
    return StmtExprVisitor::Visit_(op);
  }
};

Pass SpecializeEntryHelpers() {
  auto transform = [](IRModule mod, PassContext ctx) {
    IRModule result({}, mod->source_map, mod->attrs, mod->global_infos);
    UniqueNameSupply names(mod->functions.begin(), mod->functions.end(),
                           [](const auto& kv) { return kv.first->name_hint; });
    std::unordered_set<const GlobalVarNode*> originals;
    for (auto [gv, base] : mod->functions) {
      auto func = base.as<Function>();
      auto target = func ? func.value()->GetAttr<Target>(tvm::attr::kTarget) : std::nullopt;
      if (target && target.value()->kind->name == "cuda") {
        auto rewriter = ffi::make_object<DeviceHelperSpecializer>(
            mod, result, names, target.value(), ffi::String(), false);
        result->Add(gv, rewriter->Rewrite(func.value()));
        originals.insert(rewriter->originals.begin(), rewriter->originals.end());
      } else {
        result->Add(gv, base);
      }
    }
    std::unordered_set<const GlobalVarNode*> reachable;
    std::vector<GlobalVar> pending;
    for (auto [gv, base] : result->functions) {
      if (!originals.count(gv.get())) pending.push_back(gv);
    }
    while (!pending.empty()) {
      auto gv = pending.back();
      pending.pop_back();
      if (!reachable.insert(gv.get()).second) continue;
      auto func = result->Lookup(gv).as<Function>();
      if (!func || !func.value()->body.has_value()) continue;
      auto collector = ffi::make_object<HelperCallCollector>();
      collector->Visit(func.value()->body.value());
      for (auto callee : collector->calls) {
        if (result->functions.count(callee)) pending.push_back(callee);
      }
    }
    ffi::Array<GlobalVar> remove;
    for (auto [gv, base] : result->functions) {
      if (originals.count(gv.get()) && !reachable.count(gv.get())) remove.push_back(gv);
    }
    for (auto gv : remove) result->Remove(gv);
    return result;
  };
  return CreateModulePass(transform, 0, "cuda.SpecializeEntryHelpers");
}

Pass SpecializeDeviceHelpers() {
  auto transform = [](IRModule mod, PassContext ctx) {
    IRModule result({}, mod->source_map, mod->attrs, mod->global_infos);
    UniqueNameSupply names(mod->functions.begin(), mod->functions.end(),
                           [](const auto& kv) { return kv.first->name_hint; });
    std::map<std::string, ffi::ObjectPtr<DeviceHelperSpecializer>> groups;
    for (auto [gv, base] : mod->functions) {
      auto func = base.as<Function>();
      if (!func.has_value()) {
        result->Add(gv, base);
        continue;
      }
      auto target = func.value()->GetAttr<Target>(tvm::attr::kTarget);
      if (!target.has_value() || target.value()->kind->name != "cuda") {
        result->Add(gv, base);
        continue;
      }
      auto cc = func.value()->GetAttr<CallingConv>(tvm::attr::kCallingConv, CallingConv::kDefault);
      if (cc.value() != CallingConv::kDeviceKernelLaunch) {
        if (target.value()->host.has_value()) result->Add(gv, base);
        continue;
      }
      auto config = func.value()->GetAttr<ffi::String>("backend_config").value_or("");
      std::string key = std::string(target.value()->str()) + "\n" + std::string(config);
      auto& group = groups[key];
      if (!group)
        group =
            ffi::make_object<DeviceHelperSpecializer>(mod, result, names, target.value(), config);
      result->Add(gv, group->Rewrite(func.value()));
    }
    return result;
  };
  return CreateModulePass(transform, 0, "cuda.SpecializeDeviceHelpers");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tirx.backend.cuda.transforms.BindBackendConfig", BindBackendConfig)
      .def("tirx.backend.cuda.transforms.SpecializeDeviceHelpers", SpecializeDeviceHelpers)
      .def("tirx.backend.cuda.transforms.SpecializeEntryHelpers", SpecializeEntryHelpers);
}
}  // namespace transform
}  // namespace tirx
}  // namespace tvm
