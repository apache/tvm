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
 * \file tirx/analysis/calculate_allocated_memory.cc
 * \brief Calculate allocated memory per memory scope required by PrimFuncs.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <map>
#include <unordered_map>

#include "../../arith/int_operator.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

std::string GetStorageScope(const Var& var) {
  auto* ptr = var->ty.as<PointerTypeNode>();
  TVM_FFI_ICHECK(ptr) << "Buffer Var's type annotation must be of PointerType";
  return ptr->storage_scope;
}

/*!
 * \brief Allocation calculator for AllocBufferNode.
 */
class AllocBufferCalculator : public StmtExprVisitor {
 public:
  tvm::ffi::Map<ffi::String, int64_t> operator()(const PrimFunc& func) {
    this->VisitStmt(func->body);
    tvm::ffi::Map<ffi::String, int64_t> res;
    for (auto [k, v] : _max_size) {
      res.Set(ffi::String(k), v);
    }
    return res;
  }

 private:
  void VisitStmt_(const AllocBufferNode* op) override {
    std::string storage_scope = op->buffer.scope();
    auto search = _current_size.find(storage_scope);
    if (search == _current_size.end()) {
      _current_size[storage_scope] = 0;
      _max_size[storage_scope] = 0;
    }

    // Multiply the shape extents in the same overflow-checked way as
    // AllocBuffer::ConstantAllocationSize() (include/tvm/tirx/stmt.h). This is kept
    // as a separate loop, rather than calling ConstantAllocationSize() directly,
    // because that function's std::optional<int64_t> cannot distinguish "shape has
    // a non-constant extent" (size genuinely unknown here, contributes 0 bytes as
    // before) from "shape is constant but the element count overflows" (below,
    // rejected outright rather than silently treated as 0 bytes).
    bool is_constant_shape = true;
    int64_t num_elements = 1;
    for (const PrimExpr& e : op->buffer->shape) {
      const auto* imm = e.as<IntImmNode>();
      if (!imm) {
        is_constant_shape = false;
        break;
      }
      TVM_FFI_ICHECK_GE(imm->value, 0)
          << "Buffer " << op->buffer.name() << " in scope \"" << storage_scope
          << "\" has a negative shape extent (" << imm->value << "), which is not a valid "
          << "allocation size";
      TVM_FFI_ICHECK(!arith::WillOverflow<prim::MulNode>(num_elements, imm->value, 0,
                                                         std::numeric_limits<int64_t>::max()))
          << "Allocation shape of buffer " << op->buffer.name() << " in scope \"" << storage_scope
          << "\" has an element count that overflows int64_t";
      num_elements *= imm->value;
    }

    if (is_constant_shape) {
      int64_t bytes_per_element = static_cast<int64_t>(op->buffer->dtype.StorageBytes());
      TVM_FFI_ICHECK(!arith::WillOverflow<prim::MulNode>(num_elements, bytes_per_element, 0,
                                                         std::numeric_limits<int64_t>::max()))
          << "Allocation of buffer " << op->buffer.name() << " in scope \"" << storage_scope
          << "\" (" << num_elements << " elements of " << bytes_per_element
          << " bytes each) overflows int64_t when converted to a byte size";
      int64_t size = num_elements * bytes_per_element;

      TVM_FFI_ICHECK(!arith::WillOverflow<prim::AddNode>(_current_size[storage_scope], size, 0,
                                                         std::numeric_limits<int64_t>::max()))
          << "Accumulated allocation size for scope \"" << storage_scope
          << "\" overflows int64_t after adding buffer " << op->buffer.name();
      _current_size[storage_scope] += size;
    }
    // Else: the shape has a non-constant extent, so its byte size cannot be
    // determined here; it contributes 0 to _current_size, as before this fix.

    _max_size[storage_scope] = std::max(_current_size[storage_scope], _max_size[storage_scope]);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForNode* op) override {
    auto snapshot = _current_size;
    StmtExprVisitor::VisitStmt_(op);
    _current_size = snapshot;
  }
  void VisitStmt_(const IfThenElseNode* op) override {
    auto snapshot = _current_size;
    StmtExprVisitor::VisitStmt_(op);
    _current_size = snapshot;
  }
  void VisitStmt_(const AttrStmtNode* op) override {
    auto snapshot = _current_size;
    StmtExprVisitor::VisitStmt_(op);
    _current_size = snapshot;
  }
  std::unordered_map<std::string, int64_t> _max_size;
  std::unordered_map<std::string, int64_t> _current_size;
};

tvm::ffi::Map<ffi::String, tvm::ffi::Map<ffi::String, int64_t> > CalculateAllocatedBytes(
    const PrimFunc& func) {
  tvm::ffi::Map<ffi::String, tvm::ffi::Map<ffi::String, int64_t> > results;
  auto alloc_buffer_result = AllocBufferCalculator()(func);
  results.Set("main", alloc_buffer_result);
  return results;
}

tvm::ffi::Map<ffi::String, tvm::ffi::Map<ffi::String, int64_t> > CalculateAllocatedBytes(
    const IRModule& mod) {
  tvm::ffi::Map<ffi::String, tvm::ffi::Map<ffi::String, int64_t> > results;
  for (const auto& kv : mod->functions) {
    if (auto prim_func = kv.second.as<tirx::PrimFunc>()) {
      ffi::String func_name = kv.first->name_hint;
      auto alloc_buffer_result = AllocBufferCalculator()(prim_func.value());
      results.Set(func_name, alloc_buffer_result);
    }
  }
  return results;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "s_tir.analysis.calculate_allocated_bytes",
      [](ffi::ObjectRef obj) -> tvm::ffi::Map<ffi::String, tvm::ffi::Map<ffi::String, int64_t> > {
        if (auto func = obj.as<PrimFunc>()) {
          return CalculateAllocatedBytes(func.value());
        } else if (auto mod = obj.as<IRModule>()) {
          return CalculateAllocatedBytes(mod.value());
        } else {
          TVM_FFI_THROW(TypeError)
              << "Expect the input to be either PrimFunc or IRModule, but gets: "
              << obj->GetTypeKey();
          throw;
        }
      });
}

bool VerifyVTCMLimit(const IRModule& mod, int64_t limit) {
  auto all_sizes = CalculateAllocatedBytes(mod);
  for (const auto& kv : all_sizes) {
    auto sizes = kv.second;
    const auto vtcm_allocated = sizes.Get("global.vtcm").value_or(0);
    if (limit > 0 && vtcm_allocated > limit) {
      return false;
    }
  }
  return true;
}

bool VerifyVTCMLimit(const PrimFunc& func, int64_t limit) {
  auto sizes = CalculateAllocatedBytes(func)["main"];
  const auto vtcm_allocated = sizes.Get("global.vtcm").value_or(0);
  if (limit > 0 && vtcm_allocated > limit) {
    return false;
  }
  return true;
}

int64_t GetVTCMCapacity(Target target, const tvm::transform::PassContext& pass_ctx) {
  if (!target.defined()) target = Target::Current(/*allow_not_defined=*/true);
  if (target.defined() && target->kind->name == "hexagon") {
    auto value = target->GetAttr<int64_t>("vtcm-capacity").value();
    if (value > 0) return value;
  }
  return pass_ctx->GetConfig<int64_t>("tirx.vtcm_capacity", 0).value();
}

ffi::Array<tvm::transform::Pass> GetVTCMCompactionPasses() {
  auto pass_list = ffi::Array<tvm::transform::Pass>();
  pass_list.push_back(s_tir::transform::LowerInitBlock());
  pass_list.push_back(s_tir::transform::PlanAndUpdateBufferAllocationLocation());
  pass_list.push_back(s_tir::transform::ConvertBlocksToOpaque());
  pass_list.push_back(s_tir::transform::CompactBufferAllocation());
  pass_list.push_back(s_tir::transform::LowerMatchBuffer());
  pass_list.push_back(s_tir::transform::InjectSoftwarePipeline());
  pass_list.push_back(s_tir::transform::LowerOpaqueBlock());
  pass_list.push_back(tirx::transform::FlattenBuffer());
  pass_list.push_back(tirx::transform::StmtSimplify());
  pass_list.push_back(tirx::transform::VectorizeLoop(true));
  pass_list.push_back(tirx::transform::StorageRewrite());
  return pass_list;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.analysis.get_vtcm_compaction_passes",
                        []() { return GetVTCMCompactionPasses(); });
}

namespace transform {

Pass VerifyVTCMLimit(ffi::Optional<Target> default_target) {
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
          if (vtcm_allocated > limit.value()) {
            TVM_FFI_THROW(RuntimeError)
                << "The global.vtcm memory allocation limit has been exceeded "
                << "(allocated: " << vtcm_allocated << ", limit: " << limit.value() << ").\n"
                << "In function\n"
                << func;
          }
        }
      }
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "s_tir.VerifyVTCMLimit", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.VerifyVTCMLimit", VerifyVTCMLimit);
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
