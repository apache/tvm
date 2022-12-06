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

tvm::Map<String, Integer> CalculateAllocatedBytes(const PrimFunc& func) {
  return AllocationCalculator<AllocateNode>()(func);
}

TVM_REGISTER_GLOBAL("tir.analysis.calculate_allocated_bytes").set_body_typed([](PrimFunc func) {
  return CalculateAllocatedBytes(func);
});

bool VerifyVTCMLimit(const PrimFunc& func, Integer limit) {
  auto sizes = CalculateAllocatedBytes(func);
  const auto vtcm_allocated = sizes.Get("global.vtcm").value_or(0);
  if (limit.IntValue() > 0 && vtcm_allocated.IntValue() > limit.IntValue()) {
    return false;
  }
  return true;
}

namespace transform {

Pass VerifyVTCMLimit(const Integer& limit) {
  auto pass_func = [=](IRModule mod, PassContext ctx) {
    for (auto kv : mod->functions) {
      if (auto* n = kv.second.as<PrimFuncNode>()) {
        auto func = GetRef<PrimFunc>(n);
        auto sizes = CalculateAllocatedBytes(func);
        const auto vtcm_allocated = sizes.Get("global.vtcm").value_or(0);
        if (limit.IntValue() > 0 && vtcm_allocated.IntValue() > limit.IntValue()) {
          LOG(FATAL) << "RuntimeError: The global.vtcm memory allocation limit has been "
                        "exceeded(allocated: "
                     << vtcm_allocated << ", limit: " << limit << ").\n"
                     << "In function\n"
                     << func;
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
