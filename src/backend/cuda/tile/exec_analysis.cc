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

#include "exec_analysis.h"

#include <tvm/backend/cuda/attr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/pattern.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/annotation.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tile/context.h>
#include <tvm/tirx/tile/thread_scope.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "exec_context.h"
#include "filter_canonical.h"
namespace tvm::backend::cuda {
using namespace tvm::tirx;
namespace {
class ElectSyncFinder : public StmtExprVisitor {
 public:
  static bool Contains(const PrimExpr& expr) {
    auto finder = ffi::make_object<ElectSyncFinder>();
    finder->Visit(expr);
    return finder->found_;
  }

 private:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) final {
    auto is_canonical_elect_sync = [&]() {
      static const Op ptx_elect_sync_op = Op::Get("tirx.cuda.elect_sync");
      return op->op.same_as(ptx_elect_sync_op);
    };
    if (is_canonical_elect_sync()) {
      found_ = true;
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }

  bool found_{false};
};

class ScopeIdVarFinder : public StmtExprVisitor {
 public:
  static bool Contains(const PrimExpr& expr, const std::vector<PrimVar>& vars) {
    auto finder = ffi::make_object<ScopeIdVarFinder>(vars);
    finder->Visit(expr);
    return finder->found_;
  }

  explicit ScopeIdVarFinder(const std::vector<PrimVar>& vars) : vars_(vars) {}

 private:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) final {
    for (const PrimVar& candidate : vars_) {
      if (candidate.get() == op) {
        found_ = true;
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  const std::vector<PrimVar>& vars_;
  bool found_{false};
};

using CudaLaunchParams = std::unordered_map<ffi::String, ffi::Tuple<PrimVar, PrimExpr>>;

// CUDA indices are ordinary pure calls. Lower them after instruction dispatch so
// implementations can introduce indices without changing the launch configuration.
class CudaIndexLowerer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit CudaIndexLowerer(CudaLaunchParams params) : params_(std::move(params)) {}

  static PrimExpr Register(const std::string& name) {
    std::string operation;
    if (name.rfind("ctaid.", 0) == 0) operation = "block_idx";
    if (name.rfind("tid.", 0) == 0) operation = "thread_idx";
    if (name.rfind("ntid.", 0) == 0) operation = "block_dim";
    if (name.rfind("nctaid.", 0) == 0) operation = "grid_dim";
    if (!operation.empty()) {
      return Call(PrimType::Int(32), Op::Get("tirx.cuda." + operation),
                  {StringImm(name.substr(name.size() - 1))})
          .as_or_throw<PrimExpr>();
    }
    return Call(PrimType::Int(32), Op::Get("tirx.cuda.mov_sreg"),
                {IntImm::Int32(32), StringImm(name)})
        .as_or_throw<PrimExpr>();
  }

  static Stmt Lower(const Stmt& body, const CudaLaunchParams& params) {
    auto lowerer = ffi::make_object<CudaIndexLowerer>(params);
    for (const auto& [tag, iv] : params) {
      std::string name(tag);
      std::string reg;
      if (name.rfind("blockIdx.", 0) == 0) reg = "ctaid." + name.substr(9);
      if (name.rfind("threadIdx.", 0) == 0) reg = "tid." + name.substr(10);
      if (name.rfind("clusterCtaIdx.", 0) == 0) reg = "cluster_ctaid." + name.substr(14);
      if (!reg.empty()) lowerer->VarRemapSet(iv.get<0>(), Register(reg));
    }
    auto result = lowerer->Mutate(body).ValueOrUnchanged(body);
    ffi::Array<Stmt> prefix;
    if (lowerer->needs_warp_) {
      int64_t threads = 1;
      for (int axis = 0; axis < 3; ++axis) {
        auto extent = lowerer->BlockDimension(axis).as<IntImmNode>();
        TVM_FFI_CHECK(extent, ValueError) << "Warp indices require a static block size";
        auto dimension = extent->value.as<int64_t>();
        TVM_FFI_CHECK(dimension && *dimension > 0 && *dimension <= 1024 / threads, ValueError)
            << "Block size must be positive and contain at most 1024 threads";
        threads *= *dimension;
      }
      TVM_FFI_CHECK(threads % 32 == 0, ValueError)
          << "Warp indices require a block size divisible by 32";
      prefix.push_back(Bind(
          lowerer->warp_,
          Call(PrimType::Int(32), tirx::gpu_warp_shuffle_op(),
               {IntImm(PrimType::UInt(32), 0xffffffff), prim::FloorDiv(lowerer->LinearThread(), 32),
                IntImm::Int32(0), IntImm::Int32(32), IntImm::Int32(32)})));
    }
    prefix.push_back(result);
    return SeqStmt(prefix);
  }

 private:
  PrimExpr Axis(const std::string& tag, int axis) {
    auto it = params_.find(tag + std::string(1, 'x' + axis));
    if (it == params_.end()) return IntImm::Int32(0);
    // An explicit singleton cluster still needs a hardware index when preferred
    // clusters are enabled, so do not constant-fold cluster coordinates.
    if (tag != "clusterCtaIdx." && prim::IsOne(it->second.get<1>())) return IntImm::Int32(0);
    std::string reg = tag == "blockIdx."    ? "ctaid."
                      : tag == "threadIdx." ? "tid."
                                            : "cluster_ctaid.";
    return Register(reg + std::string(1, 'x' + axis));
  }
  PrimExpr BlockDimension(int axis) {
    auto it = params_.find("threadIdx." + std::string(1, 'x' + axis));
    if (it != params_.end() && it->second.get<1>().as<IntImmNode>()) return it->second.get<1>();
    return Register("ntid." + std::string(1, 'x' + axis));
  }
  PrimExpr LinearThread() {
    sym::Analyzer analyzer;
    return analyzer->Simplify(
        Axis("threadIdx.", 0) +
        BlockDimension(0) * (Axis("threadIdx.", 1) + BlockDimension(1) * Axis("threadIdx.", 2)));
  }
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    const auto* name_op = op->op.as<OpNode>();
    if (!name_op) return StmtExprMutator::Mutate_(op, inplace_mode);
    std::string name(name_op->name);
    if (name.rfind("tirx.cuda.", 0) != 0) return StmtExprMutator::Mutate_(op, inplace_mode);
    name = name.substr(10);
    auto axis = [&]() { return op->args[0].as_or_throw<StringImm>()->value.c_str()[0] - 'x'; };
    if (name == "block_idx") return Axis("blockIdx.", axis());
    if (name == "thread_idx") return Axis("threadIdx.", axis());
    if (name == "cluster_cta_id") {
      TVM_FFI_CHECK(params_.count("clusterCtaIdx.x"), ValueError)
          << "cluster_cta_id requires an explicit launch cluster";
      return Axis("clusterCtaIdx.", axis());
    }
    std::string reg;
    if (name == "cluster_id") reg = "clusterid.";
    if (name == "grid_dim") reg = "nctaid.";
    if (name == "block_dim") reg = "ntid.";
    if (name == "cluster_dim") reg = "cluster_nctaid.";
    if (!reg.empty()) {
      if (name == "cluster_id" || name == "cluster_dim") {
        TVM_FFI_CHECK(params_.count("clusterCtaIdx.x"), ValueError)
            << name << " requires an explicit launch cluster";
      }
      return Register(reg + std::string(1, 'x' + axis()));
    }
    if (name == "linear_thread_id") return LinearThread();
    if (name == "lane_id") return prim::FloorMod(LinearThread(), 32);
    if (name == "thread_in_warpgroup") return prim::FloorMod(LinearThread(), 128);
    if (name == "warp_id" || name == "warpgroup_id" || name == "warp_in_warpgroup") {
      needs_warp_ = true;
      if (name == "warp_id") return warp_;
      if (name == "warpgroup_id") return prim::FloorDiv(warp_, 4);
      return prim::FloorMod(warp_, 4);
    }
    if (name == "cta_pair_id") {
      TVM_FFI_CHECK(params_.count("clusterCtaIdx.x"), ValueError)
          << "cta_pair_id requires an explicit launch cluster";
      return prim::FloorMod(Axis("clusterCtaIdx.", 0) +
                                Register("cluster_nctaid.x") *
                                    (Axis("clusterCtaIdx.", 1) +
                                     Register("cluster_nctaid.y") * Axis("clusterCtaIdx.", 2)),
                            2);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }
  CudaLaunchParams params_;
  PrimVar warp_{"warp_id_in_cta", PrimType::Int(32)};
  bool needs_warp_{false};
};

}  // namespace

struct ExecAnalysis::Impl {
  Impl(const LaunchParams& params, const std::unordered_map<std::string, PrimExpr>& bounds,
       bool variable_cluster)
      : variable_cluster_(variable_cluster), cluster_bounds_(bounds), launch_params_(params) {
    PushKernelEntryCtx();
  }
  bool PushKernelEntryCtx() {
    auto prod_extent = [&](std::initializer_list<const char*> keys) -> int64_t {
      int64_t n = 1;
      for (const char* k : keys) {
        auto it = launch_params_.find(ffi::String(k));
        if (it == launch_params_.end()) continue;
        const auto* imm = it->second.get<1>().as<IntImmNode>();
        if (imm == nullptr) return 0;  // symbolic
        auto product = (n * imm->value).as<int64_t>();
        if (!product.has_value()) return 0;
        n = *product;
      }
      return n;
    };
    auto collect_extents = [&](std::initializer_list<std::pair<const char*, const char*>> keys) {
      std::vector<std::pair<std::string, int64_t>> out;
      for (const auto& [thread_key, axis_name] : keys) {
        auto it = launch_params_.find(ffi::String(thread_key));
        if (it == launch_params_.end()) continue;
        PrimExpr extent = it->second.get<1>();
        if (auto bound = cluster_bounds_.find(thread_key); bound != cluster_bounds_.end())
          extent = bound->second;
        const auto* imm = extent.as<IntImmNode>();
        if (imm == nullptr) return std::vector<std::pair<std::string, int64_t>>();
        auto value = imm->value.as<int64_t>();
        if (!value.has_value()) return std::vector<std::pair<std::string, int64_t>>();
        out.push_back({axis_name, *value});
      }
      return out;
    };
    int64_t thread_ext = prod_extent({"threadIdx.x", "threadIdx.y", "threadIdx.z"});
    if (thread_ext <= 0) {
      // launch params missing or symbolic; ExecContext tracking is not
      // available for this kernel. Dispatchers fall back to scope_kind only.
      LOG(WARNING) << "ExecContext tracking disabled: missing/symbolic threadIdx extents";
      return false;
    }
    // A sub-warp block still has one warp, with only its leading lanes active.
    int64_t lane_ext = std::min<int64_t>(thread_ext, 32);
    int64_t warp_ext = std::max<int64_t>(thread_ext / 32, 1);
    auto cluster_cta_axes = collect_extents(
        {{"clusterCtaIdx.x", "cbx"}, {"clusterCtaIdx.y", "cby"}, {"clusterCtaIdx.z", "cbz"}});
    if (!cluster_bounds_.empty() && cluster_cta_axes.empty()) return false;
    while (cluster_cta_axes.size() > 1 && cluster_cta_axes.back().second == 1)
      cluster_cta_axes.pop_back();
    cluster_cta_axis_extents_ = cluster_cta_axes;
    auto cta_axes = cluster_cta_axes;
    if (cta_axes.empty()) {
      cta_axes =
          collect_extents({{"blockIdx.x", "bx"}, {"blockIdx.y", "by"}, {"blockIdx.z", "bz"}});
      cluster_cta_axis_extents_.clear();
    }
    while (cta_axes.size() > 1 && cta_axes.back().second == 1) cta_axes.pop_back();
    int64_t cta_ext = 1;
    for (const auto& axis : cta_axes) {
      if (axis.second <= 0 || axis.second > std::numeric_limits<int64_t>::max() / cta_ext)
        return false;
      cta_ext *= axis.second;
    }
    // Preserve the old flattened cta_id split for 0-D/1-D declarations. Multi-dimensional
    // CTA ids keep their concrete factor axes (bx/by/bz or cbx/cby/cbz).
    if (cta_axes.size() <= 1) cta_axes.clear();
    ctx_stack_.push_back(ExecContext::AtKernelEntry(lane_ext, warp_ext, cta_ext, cta_axes));
    return true;
  }

  struct ScopeIdTarget {
    ScopeBinding binding;
    int dim = 0;
    int ndim = 1;
  };

  struct ScopeIdRange {
    ScopeIdTarget target;
    int64_t lo = sym::ConstIntBound::kNegInf;
    int64_t hi = sym::ConstIntBound::kPosInf;
  };

  struct PendingRangeGroup {
    ScopeIdTarget target;
    int64_t lo = sym::ConstIntBound::kNegInf;
    int64_t hi = sym::ConstIntBound::kPosInf;
    std::vector<size_t> indices;
  };

  static bool SameScopeIdTarget(const ScopeIdTarget& lhs, const ScopeIdTarget& rhs) {
    return lhs.binding == rhs.binding && lhs.dim == rhs.dim && lhs.ndim == rhs.ndim;
  }

  bool KernelCtaPredicateOverlapsClusterCta(const ScopeIdTarget& target) const {
    return target.binding == ScopeBinding::kKernelCta && !cluster_cta_axis_extents_.empty();
  }

  std::optional<ScopeIdTarget> ResolveScopeIdTarget(const PrimExpr& expr) const {
    if (auto cast = expr.as<prim::CastNode>()) return ResolveScopeIdTarget(cast->value);
    if (auto call = expr.as<CallNode>()) {
      const auto* op = call->op.as<OpNode>();
      if (!op) return std::nullopt;
      std::string name(op->name);
      ScopeBinding binding;
      std::string tag;
      if (name == "tirx.cuda.block_idx") {
        binding = ScopeBinding::kKernelCta;
        tag = "blockIdx.";
      } else if (name == "tirx.cuda.cluster_cta_id") {
        binding = ScopeBinding::kClusterCta;
        tag = "clusterCtaIdx.";
      } else if (name == "tirx.cuda.thread_idx") {
        binding = ScopeBinding::kCtaThread;
        tag = "threadIdx.";
      } else if (name == "tirx.cuda.cluster_id")
        return ScopeIdTarget{ScopeBinding::kKernelCluster, 0, 3};
      else if (name == "tirx.cuda.warp_id")
        return ScopeIdTarget{ScopeBinding::kCtaWarp};
      else if (name == "tirx.cuda.warpgroup_id")
        return ScopeIdTarget{ScopeBinding::kCtaWarpgroup};
      else if (name == "tirx.cuda.warp_in_warpgroup")
        return ScopeIdTarget{ScopeBinding::kWarpgroupWarp};
      else if (name == "tirx.cuda.lane_id")
        return ScopeIdTarget{ScopeBinding::kWarpThread};
      else if (name == "tirx.cuda.thread_in_warpgroup")
        return ScopeIdTarget{ScopeBinding::kWarpgroupThread};
      else if (name == "tirx.cuda.cta_pair_id")
        return ScopeIdTarget{ScopeBinding::kClusterCtaPair};
      else if (name == "tirx.cuda.linear_thread_id")
        return ScopeIdTarget{ScopeBinding::kCtaThread};
      else
        return std::nullopt;
      int dim = call->args[0].as_or_throw<StringImm>()->value.c_str()[0] - 'x';
      int ndim = 1;
      for (int i = 1; i < 3; ++i) {
        auto key = tag + std::string(1, 'x' + i);
        auto it = launch_params_.find(key);
        if (it != launch_params_.end()) {
          auto bound = cluster_bounds_.find(key);
          const auto& extent = bound == cluster_bounds_.end() ? it->second.get<1>() : bound->second;
          if (!prim::IsOne(extent)) ndim = i + 1;
        }
      }
      return ScopeIdTarget{binding, dim, std::max(ndim, dim + 1)};
    }
    const auto* var_node = expr.as<VarNode>();
    if (var_node == nullptr) return std::nullopt;
    Var var = ffi::GetRef<Var>(var_node);
    if (auto it = scope_aliases_.find(var); it != scope_aliases_.end()) return it->second;
    return std::nullopt;
  }

  bool TryPushRangeForTarget(const ScopeIdTarget& target, int64_t lo, int64_t hi) {
    if (ctx_stack_.empty()) return false;
    if (target.binding == ScopeBinding::kClusterCtaPair) {
      if (variable_cluster_) return false;
      if (hi != lo + 1 || lo < 0 || lo > 1) return false;
      return TryPushCtaPairValue(lo);
    }
    if (KernelCtaPredicateOverlapsClusterCta(target)) return false;
    ExecContext new_ctx = ctx_stack_.back();
    std::string err;
    if (target.ndim != 1) {
      auto cta_axis = CtaAxisName(target);
      if (!cta_axis) return false;
      if (!ctx_stack_.back().WithCtaAxisFilter(*cta_axis, lo, hi, &new_ctx, &err)) return false;
      ctx_stack_.push_back(new_ctx);
      return true;
    }
    if (!ctx_stack_.back().WithFilter(target.binding, lo, hi, &new_ctx, &err)) return false;
    ctx_stack_.push_back(new_ctx);
    return true;
  }

  bool TryPushModuloForTarget(const ScopeIdTarget& target, int64_t modulus, int64_t residue) {
    if (ctx_stack_.empty()) return false;
    if (target.binding == ScopeBinding::kClusterCtaPair) return false;
    if (KernelCtaPredicateOverlapsClusterCta(target)) return false;
    ExecContext new_ctx = ctx_stack_.back();
    std::string err;
    if (target.ndim != 1) {
      auto cta_axis = CtaAxisName(target);
      if (!cta_axis) return false;
      if (!ctx_stack_.back().WithCtaAxisModulo(*cta_axis, modulus, residue, &new_ctx, &err)) {
        return false;
      }
      ctx_stack_.push_back(new_ctx);
      return true;
    }
    if (target.binding == ScopeBinding::kKernelCta || target.binding == ScopeBinding::kClusterCta) {
      if (!ctx_stack_.back().WithCtaAxisModulo("cta_id", modulus, residue, &new_ctx, &err)) {
        return false;
      }
      ctx_stack_.push_back(new_ctx);
      return true;
    }
    return false;
  }

  bool TryPushCtaPairValue(int64_t value) {
    if (ctx_stack_.empty()) return false;
    if (cluster_cta_axis_extents_.empty()) return false;
    if (cluster_cta_axis_extents_.size() <= 1) {
      ExecContext new_ctx = ctx_stack_.back();
      std::string err;
      if (!ctx_stack_.back().WithCtaAxisModulo("cta_id", 2, value, &new_ctx, &err)) return false;
      ctx_stack_.push_back(new_ctx);
      return true;
    }

    std::optional<std::string> parity_axis;
    int64_t coeff = 1;
    int64_t fixed = 0;
    for (const auto& [axis, extent] : cluster_cta_axis_extents_) {
      AxisRange range;
      if (!ctx_stack_.back().A.GetAxis(axis, &range)) return false;
      int64_t active_extent = 0;
      int64_t active_offset = 0;
      int64_t active_stride = 0;
      if (!TryExtractIntImm(range.extent, &active_extent) ||
          !TryExtractIntImm(range.offset, &active_offset) ||
          !TryExtractIntImm(range.stride, &active_stride)) {
        return false;
      }
      fixed += coeff * active_offset;
      if (active_extent > 1 && (coeff * active_stride) % 2 != 0) {
        if (parity_axis) return false;
        parity_axis = axis;
      }
      coeff *= extent;
    }
    int64_t residue = (value - fixed) % 2;
    if (residue < 0) residue += 2;
    if (!parity_axis) {
      if (residue != 0) return false;
      ctx_stack_.push_back(ctx_stack_.back());
      return true;
    }

    ExecContext new_ctx = ctx_stack_.back();
    std::string err;
    if (!ctx_stack_.back().WithCtaAxisModulo(*parity_axis, 2, residue, &new_ctx, &err)) {
      return false;
    }
    ctx_stack_.push_back(new_ctx);
    return true;
  }

  static std::optional<std::string> CtaAxisName(const ScopeIdTarget& target) {
    static constexpr const char* kKernelCtaAxes[] = {"bx", "by", "bz"};
    static constexpr const char* kClusterCtaAxes[] = {"cbx", "cby", "cbz"};
    if (target.dim < 0 || target.dim >= 3) return std::nullopt;
    if (target.binding == ScopeBinding::kKernelCta) {
      return std::string(kKernelCtaAxes[target.dim]);
    }
    if (target.binding == ScopeBinding::kClusterCta) {
      return std::string(kClusterCtaAxes[target.dim]);
    }
    return std::nullopt;
  }

  bool TryPushSelectorForTarget(const ScopeIdTarget& target, PrimExpr selector) {
    if (ctx_stack_.empty()) return false;
    if (target.ndim != 1) return false;
    if (KernelCtaPredicateOverlapsClusterCta(target)) return false;
    ExecContext new_ctx = ctx_stack_.back();
    std::string err;
    if (!ctx_stack_.back().WithSelector(target.binding, selector, &new_ctx, &err)) return false;
    ctx_stack_.push_back(new_ctx);
    return true;
  }

  static bool TryExtractIntImm(const PrimExpr& expr, int64_t* value) {
    if (const auto* imm = expr.as<IntImmNode>()) {
      if (auto value_i64 = imm->value.as<int64_t>(); value_i64.has_value()) {
        *value = *value_i64;
        return true;
      }
    }
    return false;
  }

  std::vector<std::pair<PrimVar, ScopeIdTarget>> ScopeIdTargets() const {
    std::vector<std::pair<PrimVar, ScopeIdTarget>> out;
    for (const auto& [var, target] : scope_aliases_)
      out.push_back({var.as_or_throw<PrimVar>(), target});
    return out;
  }

  std::vector<PrimVar> ScopeIdVars() const {
    std::vector<PrimVar> vars;
    for (const auto& [var, _] : ScopeIdTargets()) {
      vars.push_back(var);
    }
    return vars;
  }

  bool ContainsScopeIdVar(const PrimExpr& pred) const {
    return ScopeIdVarFinder::Contains(pred, ScopeIdVars());
  }

  bool TryExtractLinearScopeDiff(const PrimExpr& diff, ScopeIdTarget* target, int64_t* coeff,
                                 int64_t* base) {
    PrimExpr simplified = analyzer_->Simplify(diff);
    for (const auto& [var, candidate] : ScopeIdTargets()) {
      ffi::Array<PrimExpr> linear = sym::DetectLinearEquation(simplified, {var});
      if (linear.size() != 2) continue;
      int64_t c = 0;
      int64_t b = 0;
      if (!TryExtractIntImm(analyzer_->Simplify(linear[0]), &c) ||
          !TryExtractIntImm(analyzer_->Simplify(linear[1]), &b)) {
        continue;
      }
      if (c != 1 && c != -1) continue;
      *target = candidate;
      *coeff = c;
      *base = b;
      return true;
    }
    return false;
  }

  bool TryExtractLinearCompareRange(const PrimExpr& lhs, const PrimExpr& rhs, bool inclusive,
                                    bool lhs_less_rhs, ScopeIdRange* range) {
    ScopeIdTarget target;
    int64_t coeff = 0;
    int64_t base = 0;
    if (!TryExtractLinearScopeDiff(lhs - rhs, &target, &coeff, &base)) return false;

    // Interpret `coeff * v + base <op> 0` where coeff is +/- 1.
    int64_t lo = sym::ConstIntBound::kNegInf;
    int64_t hi = sym::ConstIntBound::kPosInf;
    if (lhs_less_rhs) {
      if (coeff == 1) {
        // v + base < 0  -> v < -base
        // v + base <= 0 -> v <= -base
        hi = inclusive ? -base + 1 : -base;
      } else {
        // -v + base < 0  -> v > base
        // -v + base <= 0 -> v >= base
        lo = inclusive ? base : base + 1;
      }
    } else {
      if (coeff == 1) {
        // v + base > 0  -> v > -base
        // v + base >= 0 -> v >= -base
        lo = inclusive ? -base : -base + 1;
      } else {
        // -v + base > 0  -> v < base
        // -v + base >= 0 -> v <= base
        hi = inclusive ? base + 1 : base;
      }
    }
    *range = ScopeIdRange{target, lo, hi};
    return true;
  }

  bool TryPushLinearCompare(const PrimExpr& lhs, const PrimExpr& rhs, bool inclusive,
                            bool lhs_less_rhs) {
    ScopeIdRange range;
    if (!TryExtractLinearCompareRange(lhs, rhs, inclusive, lhs_less_rhs, &range)) return false;
    return TryPushRangeForTarget(range.target, range.lo, range.hi);
  }

  bool TryExtractLinearEqualityRange(const PrimExpr& lhs, const PrimExpr& rhs,
                                     ScopeIdRange* range) {
    ScopeIdTarget target;
    int64_t coeff = 0;
    int64_t base = 0;
    if (!TryExtractLinearScopeDiff(lhs - rhs, &target, &coeff, &base)) return false;
    int64_t value = (coeff == 1) ? -base : base;
    *range = ScopeIdRange{target, value, value + 1};
    return true;
  }

  bool TryPushLinearEquality(const PrimExpr& lhs, const PrimExpr& rhs) {
    ScopeIdRange range;
    if (!TryExtractLinearEqualityRange(lhs, rhs, &range)) return false;
    return TryPushRangeForTarget(range.target, range.lo, range.hi);
  }

  bool TryExtractModuloTarget(const PrimExpr& expr, ScopeIdTarget* target, int64_t* modulus) {
    PrimExpr lhs{ffi::UnsafeInit{}};
    PrimExpr rhs{ffi::UnsafeInit{}};
    if (const auto* mod = expr.as<prim::ModNode>()) {
      lhs = mod->a;
      rhs = mod->b;
    } else if (const auto* floormod = expr.as<prim::FloorModNode>()) {
      lhs = floormod->a;
      rhs = floormod->b;
    } else {
      return false;
    }
    auto maybe_target = ResolveScopeIdTarget(lhs);
    if (!maybe_target) return false;
    int64_t mod_value = 0;
    if (!TryExtractIntImm(analyzer_->Simplify(rhs), &mod_value) || mod_value <= 0) return false;
    *target = *maybe_target;
    *modulus = mod_value;
    return true;
  }

  bool TryPushModuloEquality(const PrimExpr& lhs, const PrimExpr& rhs) {
    ScopeIdTarget target;
    int64_t modulus = 0;
    int64_t residue = 0;
    if (TryExtractModuloTarget(lhs, &target, &modulus) &&
        TryExtractIntImm(analyzer_->Simplify(rhs), &residue)) {
      return TryPushModuloForTarget(target, modulus, residue);
    }
    if (TryExtractModuloTarget(rhs, &target, &modulus) &&
        TryExtractIntImm(analyzer_->Simplify(lhs), &residue)) {
      return TryPushModuloForTarget(target, modulus, residue);
    }
    return false;
  }

  bool TryPushComparisonPredicate(const PrimExpr& pred) {
    if (const auto* eq = pred.as<prim::EQNode>()) {
      return TryPushLinearEquality(eq->a, eq->b) || TryPushModuloEquality(eq->a, eq->b);
    }
    if (const auto* lt = pred.as<prim::LTNode>()) {
      return TryPushLinearCompare(lt->a, lt->b, /*inclusive=*/false, /*lhs_less_rhs=*/true);
    }
    if (const auto* le = pred.as<prim::LENode>()) {
      return TryPushLinearCompare(le->a, le->b, /*inclusive=*/true, /*lhs_less_rhs=*/true);
    }
    if (const auto* gt = pred.as<prim::GTNode>()) {
      return TryPushLinearCompare(gt->a, gt->b, /*inclusive=*/false, /*lhs_less_rhs=*/false);
    }
    if (const auto* ge = pred.as<prim::GENode>()) {
      return TryPushLinearCompare(ge->a, ge->b, /*inclusive=*/true, /*lhs_less_rhs=*/false);
    }
    return false;
  }

  bool TryExtractComparisonRange(const PrimExpr& pred, ScopeIdRange* range) {
    if (const auto* eq = pred.as<prim::EQNode>()) {
      return TryExtractLinearEqualityRange(eq->a, eq->b, range);
    }
    if (const auto* lt = pred.as<prim::LTNode>()) {
      return TryExtractLinearCompareRange(lt->a, lt->b, /*inclusive=*/false,
                                          /*lhs_less_rhs=*/true, range);
    }
    if (const auto* le = pred.as<prim::LENode>()) {
      return TryExtractLinearCompareRange(le->a, le->b, /*inclusive=*/true,
                                          /*lhs_less_rhs=*/true, range);
    }
    if (const auto* gt = pred.as<prim::GTNode>()) {
      return TryExtractLinearCompareRange(gt->a, gt->b, /*inclusive=*/false,
                                          /*lhs_less_rhs=*/false, range);
    }
    if (const auto* ge = pred.as<prim::GENode>()) {
      return TryExtractLinearCompareRange(ge->a, ge->b, /*inclusive=*/true,
                                          /*lhs_less_rhs=*/false, range);
    }
    return false;
  }

  void FlattenConjuncts(const PrimExpr& pred, std::vector<PrimExpr>* out) const {
    if (const auto* and_node = pred.as<prim::AndNode>()) {
      FlattenConjuncts(and_node->a, out);
      FlattenConjuncts(and_node->b, out);
      return;
    }
    if (const auto* and_node = pred.as<prim::BitwiseAndNode>()) {
      FlattenConjuncts(and_node->a, out);
      FlattenConjuncts(and_node->b, out);
      return;
    }
    out->push_back(pred);
  }

  int PushFilterPredicateCtx(const CallNode* call) {
    TVM_FFI_ICHECK_EQ(call->args.size(), 2)
        << "TIRxError: tirx.gpu_thread_filter expects (var, cond); got " << call->args.size()
        << " args";
    PrimExpr var = call->args[0].as_or_throw<PrimExpr>();
    PrimExpr cond = call->args[1].as_or_throw<PrimExpr>();
    auto target = ResolveScopeIdTarget(var);
    if (target && ElectSyncFinder::Contains(cond)) {
      PrimExpr selector = Call(var.ty(), tirx::gpu_active_thread_selector_op(), {var, cond})
                              .as_or_throw<PrimExpr>();
      int pushed = TryPushSelectorForTarget(*target, selector) ? 1 : 0;
      return pushed + PushPredicateCtx(cond);
    }
    return PushPredicateCtx(cond);
  }

  int PushConjunctivePredicateCtx(const PrimExpr& pred) {
    std::vector<PrimExpr> terms;
    FlattenConjuncts(pred, &terms);
    std::vector<bool> consumed(terms.size(), false);
    std::vector<PendingRangeGroup> groups;
    std::vector<int> term_to_group(terms.size(), -1);

    for (size_t i = 0; i < terms.size(); ++i) {
      ScopeIdRange range;
      if (!TryExtractComparisonRange(terms[i], &range)) continue;
      bool found = false;
      for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        PendingRangeGroup& group = groups[group_index];
        if (!SameScopeIdTarget(group.target, range.target)) continue;
        group.lo = std::max(group.lo, range.lo);
        group.hi = std::min(group.hi, range.hi);
        group.indices.push_back(i);
        term_to_group[i] = static_cast<int>(group_index);
        found = true;
        break;
      }
      if (!found) {
        groups.push_back(PendingRangeGroup{range.target, range.lo, range.hi, {i}});
        term_to_group[i] = static_cast<int>(groups.size() - 1);
      }
    }

    int pushed = 0;
    bool progress = true;
    while (progress) {
      progress = false;
      for (size_t i = 0; i < terms.size(); ++i) {
        if (consumed[i]) continue;
        int group_index = term_to_group[i];
        if (group_index >= 0) {
          const PendingRangeGroup& group = groups[group_index];
          if (group.indices.size() > 1 && group.indices.front() != i) continue;
          if (group.lo >= group.hi) continue;
          if (TryPushRangeForTarget(group.target, group.lo, group.hi)) {
            for (size_t index : group.indices) {
              consumed[index] = true;
            }
            ++pushed;
            progress = true;
          }
          continue;
        }
        if (TryPushComparisonPredicate(terms[i])) {
          consumed[i] = true;
          ++pushed;
          progress = true;
        }
      }
    }

    for (size_t i = 0; i < terms.size(); ++i) {
      if (consumed[i]) continue;
      int group_index = term_to_group[i];
      if (group_index >= 0) {
        consumed[i] = true;
        continue;
      }
      pushed += PushPredicateCtx(terms[i]);
    }
    return pushed;
  }

  // Try to classify `cond` as a canonical thread-filter predicate
  // (see filter_canonical.h) and narrow the ExecContext on each atom.
  //
  // Range atoms that share a ScopeIdTarget are intersected into a single
  // merged range before being pushed (this mirrors PushConjunctivePredicateCtx
  // and matters for multi-axis targets like kCtaThread, where pushing the two
  // half-bounded ranges of e.g. `0 <= tid AND tid < 128` separately would
  // overflow inside NarrowFlatProductRange).
  //
  // If the predicate is not canonical but contains a `ptx_elect_sync()` call,
  // it is treated as a lane-scope thread filter with the whole predicate
  // preserved verbatim as the selector argument -- mirroring the legacy
  // PushFilterPredicateCtx behavior for forms like `elect_sync() != 0` or
  // `not elect_sync()`.
  //
  // Returns:
  //   -1   `cond` is not canonical and does not contain elect_sync -- caller
  //        should fall back to the legacy PushPredicateCtx dispatch (which
  //        handles tirx.gpu_thread_filter wrappers, linear shifts, modulo equality).
  //   >= 0 number of context frames pushed on `ctx_stack_` (may be 0 if all
  //        atoms were recognized but none could be narrowed -- e.g. a range
  //        target that overlaps a fixed CTA pair axis).
  int TryPushCanonicalCtx(const PrimExpr& cond) {
    if (ctx_stack_.empty()) return -1;
    ScopeIdPredicate is_scope_id = [this](const Var& v) {
      return ResolveScopeIdTarget(v.as_or_throw<PrimExpr>()).has_value();
    };
    auto canonical = TryClassifyCanonical(cond, is_scope_id);
    if (!canonical) {
      // Non-canonical fallback: any predicate containing elect_sync is
      // still a lane-scope thread filter. Push the predicate as an opaque
      // selector so downstream code-gen can reuse the existing selector
      // narrowing logic.
      if (ElectSyncFinder::Contains(cond)) {
        auto lane = FindLaneScopeVar();
        if (!lane) return -1;
        ScopeIdTarget target{ScopeBinding::kWarpThread, 0, 1};
        PrimExpr selector =
            Call((*lane)->ty, tirx::gpu_active_thread_selector_op(), ffi::Array<Expr>{*lane, cond})
                .as_or_throw<PrimExpr>();
        return TryPushSelectorForTarget(target, selector) ? 1 : 0;
      }
      return -1;
    }

    struct RangeGroup {
      ScopeIdTarget target;
      int64_t lo;
      int64_t hi;
    };
    std::vector<RangeGroup> groups;
    std::vector<const FilterAtom*> elect_atoms;
    for (const FilterAtom& atom : canonical->atoms) {
      if (atom.kind == FilterAtomKind::kElectSync) {
        elect_atoms.push_back(&atom);
        continue;
      }
      auto target = ResolveScopeIdTarget(atom.scopeid_var.value().as_or_throw<PrimExpr>());
      if (!target) continue;  // atom recognized but target not in scope
      bool merged = false;
      for (auto& g : groups) {
        if (!SameScopeIdTarget(g.target, *target)) continue;
        g.lo = std::max(g.lo, atom.lo);
        g.hi = std::min(g.hi, atom.hi);
        merged = true;
        break;
      }
      if (!merged) groups.push_back({*target, atom.lo, atom.hi});
    }

    // Iterative push with progress: some pushes depend on a prior push
    // (e.g. a flat warpgroup-thread range can only narrow once wgid has
    // collapsed to a single warpgroup via an equality push). Mirrors the
    // progress loop in PushConjunctivePredicateCtx.
    std::vector<bool> consumed(groups.size(), false);
    int pushed = 0;
    bool progress = true;
    while (progress) {
      progress = false;
      for (size_t i = 0; i < groups.size(); ++i) {
        if (consumed[i]) continue;
        const auto& g = groups[i];
        if (g.lo >= g.hi) {
          consumed[i] = true;  // unsatisfiable; skip
          continue;
        }
        if (TryPushRangeForTarget(g.target, g.lo, g.hi)) {
          consumed[i] = true;
          ++pushed;
          progress = true;
        }
      }
    }
    for (const FilterAtom* atom : elect_atoms) {
      if (PushElectSyncAtom(*atom)) ++pushed;
    }
    return pushed;
  }

  bool PushElectSyncAtom(const FilterAtom& atom) {
    // Bind to lane-in-warp scope. The selector wraps the call with the lane
    // Var so downstream code generation can reuse the selector(var, pred)
    // shape produced by PushFilterPredicateCtx.
    auto lane = FindLaneScopeVar();
    if (!lane) return false;
    ScopeIdTarget target{ScopeBinding::kWarpThread, 0, 1};
    PrimExpr selector = Call((*lane)->ty, tirx::gpu_active_thread_selector_op(),
                             ffi::Array<Expr>{*lane, atom.elect_sync_call.value()})
                            .as_or_throw<PrimExpr>();
    return TryPushSelectorForTarget(target, selector);
  }

  std::optional<PrimVar> FindLaneScopeVar() const {
    for (const auto& [var, target] : scope_aliases_) {
      if (target.binding == ScopeBinding::kWarpThread) return var.as_or_throw<PrimVar>();
    }
    return std::nullopt;
  }

  int PushPredicateCtx(const PrimExpr& pred) {
    if (ctx_stack_.empty()) return 0;
    if (pred.as<prim::AndNode>() || pred.as<prim::BitwiseAndNode>()) {
      return PushConjunctivePredicateCtx(pred);
    }
    if (const auto* call = pred.as<CallNode>()) {
      if (call->op.same_as(tirx::gpu_thread_filter_op())) {
        return PushFilterPredicateCtx(call);
      }
    }
    if (TryPushComparisonPredicate(pred)) return 1;
    return 0;
  }

  PrimExpr RewriteFilterCall(const CallNode* call) const {
    TVM_FFI_ICHECK_EQ(call->args.size(), 2)
        << "TIRxError: tirx.gpu_thread_filter expects (var, cond); got " << call->args.size()
        << " args";
    return AsBool(call->args[1].as_or_throw<PrimExpr>());
  }

  PrimExpr RewriteFilterCalls(const PrimExpr& pred) const {
    if (const auto* and_node = pred.as<prim::AndNode>()) {
      PrimExpr a = RewriteFilterCalls(and_node->a);
      PrimExpr b = RewriteFilterCalls(and_node->b);
      if (a.same_as(and_node->a) && b.same_as(and_node->b)) {
        return pred;
      }
      return PrimExpr(a && b);
    }
    if (const auto* op = pred.as<prim::LShiftNode>()) {
      PrimExpr a = RewriteFilterCalls(op->a);
      PrimExpr b = RewriteFilterCalls(op->b);
      if (a.same_as(op->a) && b.same_as(op->b)) return pred;
      return tvm::left_shift(a, b, op->loc);
    }
    if (const auto* op = pred.as<prim::RShiftNode>()) {
      PrimExpr a = RewriteFilterCalls(op->a);
      PrimExpr b = RewriteFilterCalls(op->b);
      if (a.same_as(op->a) && b.same_as(op->b)) return pred;
      return tvm::right_shift(a, b, op->loc);
    }
    if (const auto* op = pred.as<prim::BitwiseAndNode>()) {
      PrimExpr a = RewriteFilterCalls(op->a);
      PrimExpr b = RewriteFilterCalls(op->b);
      if (a.same_as(op->a) && b.same_as(op->b)) return pred;
      return tvm::bitwise_and(a, b, op->loc);
    }
    if (const auto* op = pred.as<prim::BitwiseOrNode>()) {
      PrimExpr a = RewriteFilterCalls(op->a);
      PrimExpr b = RewriteFilterCalls(op->b);
      if (a.same_as(op->a) && b.same_as(op->b)) return pred;
      return tvm::bitwise_or(a, b, op->loc);
    }
    if (const auto* op = pred.as<prim::BitwiseXorNode>()) {
      PrimExpr a = RewriteFilterCalls(op->a);
      PrimExpr b = RewriteFilterCalls(op->b);
      if (a.same_as(op->a) && b.same_as(op->b)) return pred;
      return tvm::bitwise_xor(a, b, op->loc);
    }
    if (const auto* op = pred.as<prim::BitwiseNotNode>()) {
      PrimExpr a = RewriteFilterCalls(op->a);
      if (a.same_as(op->a)) return pred;
      return prim::BitwiseNot(a, op->loc);
    }
    if (const auto* call = pred.as<CallNode>()) {
      if (call->op.same_as(tirx::gpu_thread_filter_op())) {
        return RewriteFilterCalls(RewriteFilterCall(call));
      }
      bool changed = false;
      ffi::Array<Expr> args;
      args.reserve(call->args.size());
      for (const Expr& arg : call->args) {
        Expr new_arg = arg;
        if (auto prim_arg = arg.as<PrimExpr>()) {
          new_arg = RewriteFilterCalls(prim_arg.value());
        }
        changed = changed || !new_arg.same_as(arg);
        args.push_back(new_arg);
      }
      if (changed) {
        return Call(call->ty, call->op, args, call->attrs, {}, call->loc).as_or_throw<PrimExpr>();
      }
    }
    return pred;
  }

  PrimExpr AsBool(PrimExpr pred) const {
    PrimType pred_ty = pred.ty();
    if (pred_ty.MatchesCode(DLDataTypeCode::kDLBool)) {
      return pred;
    }
    return pred != IntImm(pred.ty(), 0);
  }

  bool variable_cluster_;
  std::unordered_map<std::string, PrimExpr> cluster_bounds_;
  std::unordered_map<Var, ScopeIdTarget, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> scope_aliases_;
  sym::Analyzer analyzer_;
  std::vector<ExecContext> ctx_stack_;
  LaunchParams launch_params_;
  std::vector<std::pair<std::string, int64_t>> cluster_cta_axis_extents_;
};
ExecAnalysis::ExecAnalysis(const LaunchParams& params,
                           const std::unordered_map<std::string, PrimExpr>& bounds,
                           bool variable_cluster)
    : impl_(std::make_unique<Impl>(params, bounds, variable_cluster)) {}
ExecAnalysis::~ExecAnalysis() = default;
void ExecAnalysis::Bind(const Var& var, const PrimExpr& value) {
  if (auto target = impl_->ResolveScopeIdTarget(value))
    impl_->scope_aliases_.insert_or_assign(var, *target);
}
int ExecAnalysis::PushPredicate(const PrimExpr& condition) {
  int count = impl_->TryPushCanonicalCtx(condition);
  return count < 0 ? impl_->PushPredicateCtx(condition) : count;
}
void ExecAnalysis::Pop(int count) {
  while (count-- > 0) impl_->ctx_stack_.pop_back();
}
PrimExpr ExecAnalysis::RewritePredicate(const PrimExpr& condition) const {
  return impl_->RewriteFilterCalls(condition);
}
std::pair<SplitMap, SplitMap> ExecAnalysis::Split(const tirx::ThreadScope& scope) const {
  TVM_FFI_CHECK(scope->backend() == "cuda", ValueError) << "Expected a CUDA thread scope";
  TVM_FFI_CHECK(!impl_->variable_cluster_ || scope->local_name() != "cluster", ValueError)
      << "Cluster tensor instructions require one static cluster shape; "
         "use explicit CUDA/PTX instructions for variable or preferred cluster shapes";
  if (impl_->ctx_stack_.empty()) return {};
  ExecSplit split;
  std::string err;
  if (ScopeSwitch(impl_->ctx_stack_.back().A, StringToScopeKind(scope->local_name()), &split,
                  &err)) {
    return {EncodeSplitSide(split.inter), EncodeSplitSide(split.intra)};
  }
  LOG(WARNING) << "ExecContext scope_switch failed: " << err;
  return {};
}
Stmt LowerIndices(const Stmt& stmt, const LaunchParams& params) {
  return CudaIndexLowerer::Lower(stmt, params);
}
}  // namespace tvm::backend::cuda
