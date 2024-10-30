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
 * \file tir/analysis/calculate_allocated_memory.cc
 * \brief Calculate allocated memory per memory scope required by PrimFuncs.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/usmp/utils.h>

#include <algorithm>
#include <map>
#include <unordered_map>

namespace tvm {
namespace tir {

template <typename T>
class AllocationCalculator : public StmtExprVisitor {
 public:
  AllocationCalculator() = default;
  tvm::Map<String, Integer> operator()(const PrimFunc& func);

 private:
  void VisitStmt_(const T* op) override;
  std::unordered_map<std::string, int64_t> _max_size;
  std::unordered_map<std::string, int64_t> _current_size;
};

template <typename T>
tvm::Map<String, Integer> AllocationCalculator<T>::operator()(const PrimFunc& func) {
  this->VisitStmt(func->body);
  tvm::Map<String, Integer> res;
  for (auto [k, v] : _max_size) {
    res.Set(String(k), Integer(v));
  }
  return res;
}

std::string GetStorageScope(const Var& var) {
  auto* ptr = var->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr) << "Buffer Var's type annotation must be of PointerType";
  return ptr->storage_scope;
}

template <typename T>
void AllocationCalculator<T>::VisitStmt_(const T* op) {
  std::string storage_scope = GetStorageScope(op->buffer_var);
  auto search = _current_size.find(storage_scope);
  if (search == _current_size.end()) {
    _current_size[storage_scope] = 0;
    _max_size[storage_scope] = 0;
  }
  auto size = op->ConstantAllocationSize() * op->dtype.bytes() * op->dtype.lanes();
  _current_size[storage_scope] += size;
  _max_size[storage_scope] = std::max(_current_size[storage_scope], _max_size[storage_scope]);
  StmtExprVisitor::VisitStmt(op->body);
  _current_size[storage_scope] -= size;
}

tvm::Map<String, tvm::Map<String, Integer> > CalculateAllocatedBytes(const PrimFunc& func) {
  tvm::Map<String, tvm::Map<String, Integer> > results;
  results.Set("main", AllocationCalculator<AllocateNode>()(func));
  return results;
}

tvm::Map<String, tvm::Map<String, Integer> > CalculateAllocatedBytes(const IRModule& mod) {
  tvm::Map<String, tvm::Map<String, Integer> > results;
  for (const auto& kv : mod->functions) {
    if (auto prim_func = kv.second.as<tir::PrimFunc>()) {
      String func_name = kv.first->name_hint;
      results.Set(func_name, AllocationCalculator<AllocateNode>()(prim_func.value()));
    }
  }
  return results;
}

TVM_REGISTER_GLOBAL("tir.analysis.calculate_allocated_bytes")
    .set_body_typed([](ObjectRef obj) -> tvm::Map<String, tvm::Map<String, Integer> > {
      if (auto func = obj.as<PrimFunc>()) {
        return CalculateAllocatedBytes(func.value());
      } else if (auto mod = obj.as<IRModule>()) {
        return CalculateAllocatedBytes(mod.value());
      } else {
        LOG(FATAL) << "TypeError: Expect the input to be either PrimFunc or IRModule, but gets: "
                   << obj->GetTypeKey();
        throw;
      }
    });

bool VerifyVTCMLimit(const IRModule& mod, Integer limit) {
  auto all_sizes = CalculateAllocatedBytes(mod);
  for (const auto& kv : all_sizes) {
    auto sizes = kv.second;
    const auto vtcm_allocated = sizes.Get("global.vtcm").value_or(0);
    if (limit.IntValue() > 0 && vtcm_allocated.IntValue() > limit.IntValue()) {
      return false;
    }
  }
  return true;
}

bool VerifyVTCMLimit(const PrimFunc& func, Integer limit) {
  auto sizes = CalculateAllocatedBytes(func)["main"];
  const auto vtcm_allocated = sizes.Get("global.vtcm").value_or(0);
  if (limit.IntValue() > 0 && vtcm_allocated.IntValue() > limit.IntValue()) {
    return false;
  }
  return true;
}

int64_t GetVTCMCapacity(Target target, const transform::PassContext& pass_ctx) {
  if (!target.defined()) target = Target::Current(/*allow_not_defined=*/true);
  if (target.defined() && target->kind->name == "hexagon") {
    auto value = target->GetAttr<Integer>("vtcm-capacity").value()->value;
    if (value > 0) return value;
  }
  return pass_ctx->GetConfig<Integer>("tir.vtcm_capacity", Integer(0)).value()->value;
}

Array<tvm::transform::Pass> GetVTCMCompactionPasses() {
  auto pass_list = Array<tvm::transform::Pass>();
  pass_list.push_back(tir::transform::LowerInitBlock());
  pass_list.push_back(tir::transform::PlanAndUpdateBufferAllocationLocation());
  pass_list.push_back(tir::transform::ConvertBlocksToOpaque());
  pass_list.push_back(tir::transform::CompactBufferAllocation());
  pass_list.push_back(tir::transform::LowerMatchBuffer());
  pass_list.push_back(tir::transform::InjectSoftwarePipeline());
  pass_list.push_back(tir::transform::LowerOpaqueBlock());
  pass_list.push_back(tir::transform::FlattenBuffer());
  pass_list.push_back(tir::transform::Simplify());
  pass_list.push_back(tir::transform::VectorizeLoop(true));
  pass_list.push_back(tir::transform::StorageRewrite());
  return pass_list;
}

TVM_REGISTER_GLOBAL("tir.analysis.get_vtcm_compaction_passes").set_body_typed([]() {
  return GetVTCMCompactionPasses();
});

namespace transform {

Pass VerifyVTCMLimit(Optional<Target> default_target) {
  auto pass_func = [=](IRModule mod, PassContext ctx) {
    for (auto kv : mod->functions) {
      if (auto opt = kv.second.as<PrimFunc>()) {
        auto func = opt.value();

        std::optional<int64_t> limit = std::nullopt;
        if (auto func_target = func->GetAttr<Target>(tvm::attr::kTarget)) {
          limit = GetVTCMCapacity(func_target.value(), ctx);
        } else if (default_target) {
          limit = GetVTCMCapacity(default_target.value(), ctx);
        }

        if (limit.has_value() && limit.value() > 0) {
          auto sizes = CalculateAllocatedBytes(func)["main"];
          const auto vtcm_allocated = sizes.Get("global.vtcm").value_or(0);
          if (vtcm_allocated.IntValue() > limit.value()) {
            LOG(FATAL) << "RuntimeError: The global.vtcm memory allocation limit has been exceeded "
                       << "(allocated: " << vtcm_allocated << ", limit: " << limit.value() << ").\n"
                       << "In function\n"
                       << func;
          }
        }
      }
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.calculate_allocated_bytes", {});
}

TVM_REGISTER_GLOBAL("tir.transform.VerifyVTCMLimit").set_body_typed(VerifyVTCMLimit);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
